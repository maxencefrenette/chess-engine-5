use std::{num::NonZeroUsize, path::PathBuf, time::Instant};

use candle_core::{DType, Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap, loss, ops::log_softmax};
use chess_network::{ModelConfig, Net, checkpoint, piece_value::PieceValue};
use rand::{SeedableRng, seq::SliceRandom};
use rand_chacha::ChaCha8Rng;
use serde_json::json;

use crate::data::{self, Batch, Result};

const GENERATION_BUDGET_WEIGHT: f64 = 0.1;

#[chess_config::toml_config]
pub struct Config {
    dataset: PathBuf,
    output_dir: PathBuf,
    pub device: chess_config::DeviceConfig,
    train_positions: NonZeroUsize,
    optimizer_steps: NonZeroUsize,
    batch_size: NonZeroUsize,
    optimizer: OptimizerConfig,
    seed: u64,
    model: ModelConfig,
}

#[derive(serde::Deserialize, serde::Serialize)]
#[serde(tag = "kind", rename_all = "lowercase", deny_unknown_fields)]
enum OptimizerConfig {
    Adamw(AdamWConfig),
}

#[chess_config::toml_config]
#[derive(serde::Serialize)]
struct AdamWConfig {
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
    schedule: Schedule,
}

#[derive(serde::Deserialize, serde::Serialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
enum Schedule {
    Constant {},
    Cosine {
        warmup_fraction: f64,
        min_ratio: f64,
    },
}

impl Schedule {
    fn multiplier(&self, step: usize, steps: usize) -> f64 {
        match *self {
            Self::Constant {} => 1.0,
            Self::Cosine {
                warmup_fraction,
                min_ratio,
            } => {
                let warmup = ((steps as f64 * warmup_fraction).round() as usize).min(steps - 1);
                if step < warmup {
                    (step + 1) as f64 / warmup as f64
                } else {
                    let progress = (step - warmup) as f64 / (steps - warmup - 1).max(1) as f64;
                    min_ratio
                        + (1.0 - min_ratio) * (1.0 + (std::f64::consts::PI * progress).cos()) / 2.0
                }
            }
        }
    }
}

fn losses(
    net: &Net,
    playing_value: &PieceValue,
    batch: &Batch,
) -> candle_core::Result<[Tensor; 2]> {
    let policy = batch.inputs.logits(net)?;
    let log_probs = log_softmax(&(&policy + &batch.mask)?, 1)?;
    let policy = (&log_probs * &batch.policy)?.sum(1)?.neg()?.mean_all()?;
    Ok([
        policy,
        loss::cross_entropy(&playing_value.logits(&batch.piece_features)?, &batch.value)?,
    ])
}

pub fn run(config: Config, device: &Device) -> Result<()> {
    let Config {
        dataset,
        output_dir,
        train_positions,
        optimizer_steps: planned_steps,
        batch_size,
        optimizer: optimizer_config,
        seed,
        model,
        ..
    } = config;
    let OptimizerConfig::Adamw(adamw) = &optimizer_config;
    let (planned_steps, batch_size) = (planned_steps.get(), batch_size.get());
    if !adamw.learning_rate.is_finite() || adamw.learning_rate <= 0. {
        return Err("training optimizer.learning_rate must be positive and finite".into());
    }
    if let Schedule::Cosine {
        warmup_fraction,
        min_ratio,
    } = adamw.schedule
        && (!(0.0..1.0).contains(&warmup_fraction) || !(0.0..=1.0).contains(&min_ratio))
    {
        return Err("cosine warmup_fraction must be in [0, 1) and min_ratio in [0, 1]".into());
    }
    if output_dir.exists() {
        return Err(format!("output already exists: {}", output_dir.display()).into());
    }
    let setup_started = Instant::now();
    let data = data::load(&dataset, seed, train_positions.get())
        .map_err(|error| format!("dataset {}: {error}", dataset.display()))?;
    println!(
        "train: {} games / {} positions; test: {} games / {} positions",
        data.train_games,
        data.train.len(),
        data.test_ids.len(),
        data.test.len()
    );
    if device.is_cuda() {
        device.set_seed(seed)?;
    }
    let parameters = VarMap::new();
    let net = Net::new(
        VarBuilder::from_varmap(&parameters, DType::F32, device),
        model,
    )?;
    let playing_value = PieceValue::new(VarBuilder::from_varmap(&parameters, DType::F32, device))?;
    let mut optimizer = AdamW::new(
        parameters.all_vars(),
        ParamsAdamW {
            lr: adamw.learning_rate,
            beta1: adamw.beta1,
            beta2: adamw.beta2,
            eps: adamw.eps,
            weight_decay: adamw.weight_decay,
        },
    )?;
    let mut indices: Vec<_> = (0..data.train.len()).collect();
    let steps_per_epoch = indices.len().div_ceil(batch_size);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    device.synchronize()?;
    let training_started = Instant::now();
    let setup_seconds = training_started.duration_since(setup_started).as_secs_f64();
    let mut optimizer_steps = 0usize;
    for epoch in 1..=planned_steps.div_ceil(steps_per_epoch) {
        let epoch_started = Instant::now();
        indices.shuffle(&mut rng);
        let mut totals = [0.; 2];
        let mut epoch_positions = 0;
        for chunk in indices
            .chunks(batch_size)
            .take(planned_steps - optimizer_steps)
        {
            let batch = Batch::new(&data.train, chunk, device)?;
            let [policy, value] = losses(&net, &playing_value, &batch)?;
            let objective = (&policy + &value)?;
            optimizer.set_learning_rate(
                adamw.learning_rate * adamw.schedule.multiplier(optimizer_steps, planned_steps),
            );
            optimizer.backward_step(&objective)?;
            optimizer_steps += 1;
            epoch_positions += chunk.len();
            for (total, loss) in totals.iter_mut().zip([policy, value]) {
                *total += f64::from(loss.to_scalar::<f32>()?) * chunk.len() as f64;
            }
        }
        let [policy, value] = totals.map(|loss| loss / epoch_positions as f64);
        if !policy.is_finite() || !value.is_finite() {
            return Err("non-finite training loss".into());
        }
        println!(
            "epoch {epoch}: {epoch_positions}/{} positions; train policy {policy:.4}, WDL {value:.4}, {:.1}s",
            data.train.len(),
            epoch_started.elapsed().as_secs_f64()
        );
    }
    device.synchronize()?;
    let evaluation_started = Instant::now();
    let training_seconds = evaluation_started
        .duration_since(training_started)
        .as_secs_f64();

    // Load the trained parameters as constants so the test pass does not build a backward graph.
    let tensors: std::collections::HashMap<_, _> = parameters
        .data()
        .lock()
        .unwrap()
        .iter()
        .map(|(name, var)| (name.clone(), var.as_tensor().detach()))
        .collect();
    let builder = VarBuilder::from_tensors(tensors.clone(), DType::F32, device);
    let playing_value = PieceValue::new(builder.clone())?;
    let net = Net::new(builder, model)?;
    let mut counts = [1.; 3];
    for row in &data.train {
        counts[row.value as usize] += 1.;
    }
    let count_sum = counts.iter().sum::<f64>();
    let mut totals = [0.; 3];
    let mut value_baseline = 0.;
    let indices: Vec<_> = (0..data.test.len()).collect();
    for chunk in indices.chunks(batch_size) {
        let batch = Batch::new(&data.test, chunk, device)?;
        for (total, loss) in totals.iter_mut().zip(losses(&net, &playing_value, &batch)?) {
            *total += f64::from(loss.to_scalar::<f32>()?) * chunk.len() as f64;
        }
        totals[2] += batch.uniform_loss;
        for &row in chunk {
            value_baseline -= (counts[data.test[row].value as usize] / count_sum).ln();
        }
    }
    let test_count = data.test.len() as f64;
    let [policy, value, uniform] = totals.map(|loss| loss / test_count);
    let value_baseline = value_baseline / test_count;
    if !policy.is_finite() || !value.is_finite() {
        return Err("non-finite test loss".into());
    }
    let beats_uniform = policy < uniform;
    println!(
        "test policy {policy:.4} vs uniform {uniform:.4}; test WDL {value:.4} vs prior {value_baseline:.4}"
    );
    device.synchronize()?;
    let evaluation_seconds = evaluation_started.elapsed().as_secs_f64();

    let completed_epochs = optimizer_steps / steps_per_epoch;
    let partial_epoch_positions = (optimizer_steps % steps_per_epoch) * batch_size;
    println!(
        "epochs: {completed_epochs} complete, {partial_epoch_positions}/{} positions in a partial epoch",
        data.train.len()
    );

    let config = checkpoint::config(
        model,
        json!({
                "source": dataset,
                "seed": seed,
                "completed_epochs": completed_epochs,
                "partial_epoch_positions": partial_epoch_positions,
                "batch_size": batch_size,
                "optimizer": optimizer_config,
                "optimizer_steps": optimizer_steps,
                "setup_seconds": setup_seconds,
                "training_seconds": training_seconds,
                "evaluation_seconds": evaluation_seconds,
                "device": if device.is_cuda() { "cuda" } else { "cpu" },
                "dtype": "f32",
                "layer_norm_backward": true,
                "train_games": data.train_games,
                "train_positions": data.train.len(),
                "requested_train_positions": train_positions.get(),
                "estimated_generation_seconds": data.estimated_generation_seconds,
                "test_game_ids": data.test_ids,
                "test_positions": data.test.len(),
                "test_policy_loss": policy,
                "uniform_policy_loss": uniform,
                "beats_uniform_policy": beats_uniform,
                "test_wdl_loss": value,
                "prior_wdl_loss": value_baseline,
        }),
    );
    checkpoint::save(&output_dir, tensors, &config)?;
    let timed_seconds = training_started.elapsed().as_secs_f64();
    println!("saved {}", output_dir.display());
    println!(
        "runtime: {timed_seconds:.2}s timed (training, held-out evaluation, and save); {setup_seconds:.2}s setup excluded; {optimizer_steps} optimizer steps ({training_seconds:.2}s training, {evaluation_seconds:.2}s held-out evaluation)"
    );
    if let Some(generation) = data.estimated_generation_seconds {
        let spent = timed_seconds + GENERATION_BUDGET_WEIGHT * generation;
        println!(
            "training budget: {spent:.2}s spent ({timed_seconds:.2}s timed runtime + {GENERATION_BUDGET_WEIGHT} × {generation:.2}s estimated generation)"
        );
    } else {
        println!("training budget: unavailable because the dataset is missing generation timings");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use chess_config::TomlConfig;
    use chess_network::{INPUTS, SQUARES};

    use super::*;

    #[test]
    fn cosine_schedule_warms_up_and_reaches_its_floor() {
        let schedule = Schedule::Cosine {
            warmup_fraction: 0.05,
            min_ratio: 0.1,
        };
        assert_eq!(schedule.multiplier(0, 100), 0.2);
        assert_eq!(schedule.multiplier(4, 100), 1.0);
        assert_eq!(schedule.multiplier(5, 100), 1.0);
        assert!(schedule.multiplier(50, 100) < 1.0);
        assert!((schedule.multiplier(99, 100) - 0.1).abs() < 1e-12);
    }

    #[test]
    fn policy_loss_reaches_smolgen() -> Result<()> {
        let mut config = Config::from_repo_file("training.toml")?;
        config.model.layers = 1;
        let device = &Device::Cpu;
        let params = VarMap::new();
        let net = Net::new(
            VarBuilder::from_varmap(&params, DType::F32, device),
            config.model,
        )?;
        let board = Tensor::randn(0f32, 1., (1, SQUARES, INPUTS), device)?;
        let moves = Tensor::zeros((1, 1), DType::U32, device)?;
        let promotes = Tensor::zeros((1, 1), DType::F32, device)?;
        let policy = net.policy(&board, &moves, &moves, &promotes)?;
        let grads = policy.sum_all()?.backward()?;
        let params = params.data().lock().unwrap();
        for name in [
            "input.weight",
            "blocks.0.smolgen.compress.weight",
            "smolgen_projection.weight",
        ] {
            let grad = grads
                .get(params[name].as_tensor())
                .unwrap_or_else(|| panic!("no gradient for {name}"));
            assert!(grad.abs()?.sum_all()?.to_scalar::<f32>()? > 0.0, "{name}");
        }
        Ok(())
    }
}
