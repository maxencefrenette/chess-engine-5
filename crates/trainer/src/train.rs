use std::{num::NonZeroUsize, path::PathBuf, time::Instant};

use candle_core::{DType, Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap, loss, ops::log_softmax};
use chess_network::{ModelConfig, Net, checkpoint, piece_value::PieceValue};
use rand::{SeedableRng, seq::SliceRandom};
use rand_chacha::ChaCha8Rng;
use serde_json::json;

use crate::data::{self, Batch, Result};

#[chess_config::toml_config]
pub struct Config {
    dataset: PathBuf,
    output_dir: PathBuf,
    pub device: String,
    epochs: NonZeroUsize,
    max_steps: usize,
    batch_size: NonZeroUsize,
    adamw: AdamWConfig,
    seed: u64,
    model: ModelConfig,
}

#[chess_config::toml_config]
#[derive(serde::Serialize)]
struct AdamWConfig {
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
}

fn losses(
    net: &Net,
    playing_value: &PieceValue,
    batch: &Batch,
) -> candle_core::Result<[Tensor; 2]> {
    let policy = net.policy(
        &batch.position,
        &batch.move_index,
        &batch.promotion_index,
        &batch.promotes,
    )?;
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
        epochs,
        max_steps,
        batch_size,
        adamw,
        seed,
        model,
        ..
    } = config;
    let (epochs, batch_size) = (epochs.get(), batch_size.get());
    if !adamw.learning_rate.is_finite() || adamw.learning_rate <= 0. {
        return Err("training adamw.learning_rate must be positive and finite".into());
    }
    if output_dir.exists() {
        return Err(format!("output already exists: {}", output_dir.display()).into());
    }
    let setup_started = Instant::now();
    let data = data::load(&dataset, seed)
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
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    device.synchronize()?;
    let training_started = Instant::now();
    let setup_seconds = training_started.duration_since(setup_started).as_secs_f64();
    let mut optimizer_steps = 0usize;
    for epoch in 1..=epochs {
        let epoch_started = Instant::now();
        indices.shuffle(&mut rng);
        let mut totals = [0.; 2];
        let mut epoch_positions = 0;
        let remaining = if max_steps == 0 {
            usize::MAX
        } else {
            max_steps - optimizer_steps
        };
        for chunk in indices.chunks(batch_size).take(remaining) {
            let batch = Batch::new(&data.train, chunk, device)?;
            let [policy, value] = losses(&net, &playing_value, &batch)?;
            let objective = (&policy + &value)?;
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
            "epoch {epoch}/{epochs}: train policy {policy:.4}, WDL {value:.4}, {:.1}s",
            epoch_started.elapsed().as_secs_f64()
        );
        if optimizer_steps == max_steps {
            break;
        }
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

    let config = checkpoint::config(
        model,
        json!({
                "source": dataset,
                "seed": seed,
                "epochs": epochs,
                "batch_size": batch_size,
                "adamw": adamw,
                "optimizer_steps": optimizer_steps,
                "max_steps": max_steps,
                "setup_seconds": setup_seconds,
                "training_seconds": training_seconds,
                "evaluation_seconds": evaluation_seconds,
                "device": if device.is_cuda() { "cuda" } else { "cpu" },
                "dtype": "f32",
                "layer_norm_backward": true,
                "train_games": data.train_games,
                "train_positions": data.train.len(),
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
    Ok(())
}

#[cfg(test)]
mod tests {
    use chess_config::TomlConfig;
    use chess_network::{INPUTS, SQUARES};

    use super::*;

    #[test]
    fn policy_loss_reaches_smolgen() -> Result<()> {
        let mut config = Config::from_file(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../config/training.toml"
        ))?;
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
