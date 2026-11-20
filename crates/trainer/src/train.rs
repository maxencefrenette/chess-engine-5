use std::{fmt::Write, fs, time::Instant};

use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap, ops::log_softmax};
use chess_network::{Net, checkpoint};
use rand::{SeedableRng, seq::SliceRandom};
use rand_chacha::ChaCha8Rng;
use serde_json::json;

use crate::{
    config::{Config, OptimizerConfig},
    data::{self, Batch, Result},
};

const GENERATION_BUDGET_WEIGHT: f64 = 1.0;
const METRICS_INTERVAL: usize = 10;

struct TrainingSample {
    policy: f64,
    wdl: f64,
    lr: f64,
    elapsed: f64,
    step: usize,
    positions: usize,
    epoch: usize,
}

#[derive(Default)]
struct LossTotals {
    positions: usize,
    values: usize,
    policy_sum: f64,
    wdl_sum: f64,
}

impl LossTotals {
    fn add(&mut self, policy: f64, wdl: f64, positions: usize, values: usize) {
        self.positions += positions;
        self.values += values;
        self.policy_sum += policy * positions as f64;
        self.wdl_sum += wdl * values as f64;
    }

    fn means(&self) -> [f64; 2] {
        [
            self.policy_sum / self.positions as f64,
            self.wdl_sum / self.values.max(1) as f64,
        ]
    }
}

fn soft_value_loss(logits: &Tensor, targets: &Tensor) -> candle_core::Result<Tensor> {
    (log_softmax(logits, 1)? * targets)?
        .sum(1)?
        .neg()?
        .mean_all()
}

fn losses(logits: (Tensor, Tensor), batch: &Batch) -> candle_core::Result<[Tensor; 2]> {
    let (policy, wdl) = logits;
    let log_probs = log_softmax(&(&policy + &batch.mask)?, 1)?;
    let policy = (&log_probs * &batch.policy)?.sum(1)?.neg()?.mean_all()?;
    let value = if let Some(targets) = &batch.value {
        let wdl = match &batch.value_indices {
            Some(indices) => wdl.index_select(indices, 0)?,
            None => wdl,
        };
        soft_value_loss(&wdl, targets)?
    } else {
        Tensor::zeros((), policy.dtype(), policy.device())?
    };
    Ok([policy, value])
}

pub fn run(config: Config, device: &Device) -> Result<()> {
    let Config {
        ref output_dir,
        batch_size,
        ref optimizer,
        seed,
        value_loss_weight,
        ref curriculum,
        model,
        ..
    } = config;
    let OptimizerConfig::Adamw(adamw) = optimizer;
    let batch_size = batch_size.get();
    if curriculum.is_empty() {
        return Err("curriculum must contain at least one phase".into());
    }
    if output_dir.exists() {
        return Err(format!("output already exists: {}", output_dir.display()).into());
    }
    let setup_started = Instant::now();
    data::check_disjoint(curriculum.iter().map(|step| step.dataset.as_path()))?;
    let mut rescorer = config.value_rescoring.open()?;
    let mut datasets = curriculum
        .iter()
        .enumerate()
        .map(|(index, step)| {
            data::load(
                &step.dataset,
                &config,
                step.num_positions.get(),
                index + 1 == curriculum.len(),
                &mut rescorer,
            )
            .map_err(|error| format!("dataset {}: {error}", step.dataset.display()).into())
        })
        .collect::<Result<Vec<_>>>()?;
    println!(
        "tablebase rescoring: {} value labels probed, {} changed",
        rescorer.stats.probed, rescorer.stats.changed
    );
    let hints = data::attach_hints(&mut datasets, model.hints);
    let phase_steps = curriculum
        .iter()
        .zip(&datasets)
        .enumerate()
        .map(|(index, (step, data))| {
            step.optimizer_steps(data.train.len(), batch_size)
                .map_err(|error| format!("curriculum phase {}: {error}", index + 1).into())
        })
        .collect::<Result<Vec<_>>>()?;
    let planned_steps = phase_steps
        .iter()
        .try_fold(0usize, |total, &steps| total.checked_add(steps))
        .ok_or("total curriculum optimizer steps overflow")?;
    let held_out = datasets.last().expect("validated non-empty curriculum");
    let train_games = datasets.iter().map(|data| data.train_games).sum::<usize>();
    let train_positions = datasets.iter().map(|data| data.train.len()).sum::<usize>();
    let generation_seconds = datasets
        .iter()
        .map(|data| data.estimated_generation_seconds)
        .sum::<f64>();
    println!(
        "train: {} games / {} positions; test: {} games / {} positions",
        train_games,
        train_positions,
        held_out.test_ids.len(),
        held_out.test.len()
    );
    let mut value_labels: Vec<_> = datasets
        .iter()
        .flat_map(|data| data.value_labels_per_game.iter().copied())
        .collect();
    value_labels.sort_unstable();
    let value_label_count = value_labels.iter().sum::<usize>();
    let value_p90 = value_labels[(value_labels.len() * 9).div_ceil(10) - 1];
    println!(
        "value supervision: {value_label_count} labels across {} games; p90 {value_p90} labels/game",
        value_labels.len()
    );
    if device.is_cuda() {
        device.set_seed(seed)?;
    }
    let parameters = VarMap::new();
    let net = Net::new(
        VarBuilder::from_varmap(&parameters, DType::F32, device),
        model,
    )?;
    let reply_head = (config.reply_weight > 0.)
        .then(|| {
            crate::reply::Head::new(
                VarBuilder::from_varmap(&parameters, DType::F32, device).pp("reply"),
                model.width,
            )
        })
        .transpose()?;
    let mut optimizer = crate::packed_adamw::PackedAdamW::new(parameters.all_vars(), adamw.into())?;
    let mut average =
        crate::weight_averaging::TailAverage::new(&config.weight_averaging, planned_steps);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut hint_rng = ChaCha8Rng::seed_from_u64(seed);
    hint_rng.set_stream(1);
    let mut history = Vec::with_capacity(planned_steps.div_ceil(METRICS_INTERVAL));
    device.synchronize()?;
    let training_started = Instant::now();
    let setup_seconds = training_started.duration_since(setup_started).as_secs_f64();
    let mut optimizer_steps = 0usize;
    let mut positions_seen = 0;
    let mut window_totals = LossTotals::default();
    let mut epoch = 0;
    let mut phase_metrics = Vec::with_capacity(curriculum.len());
    let mut completed_epochs = 0;
    let mut partial_epoch_positions = 0;
    for (index, (data, phase_steps)) in datasets.iter().zip(phase_steps).enumerate() {
        let rows = &data.train;
        let mut indices: Vec<_> = (0..rows.len()).collect();
        let steps_per_epoch = rows.len().div_ceil(batch_size);
        let end_step = optimizer_steps + phase_steps;
        let phase_start = positions_seen;
        for _ in 0..phase_steps.div_ceil(steps_per_epoch) {
            epoch += 1;
            let epoch_started = Instant::now();
            indices.shuffle(&mut rng);
            let mut totals = LossTotals::default();
            for chunk in indices.chunks(batch_size).take(end_step - optimizer_steps) {
                let batch = Batch::new(
                    chunk.iter().map(|&i| &rows[i]),
                    device,
                    model.hints.dropout(),
                    &mut hint_rng,
                )?;
                let (policy, wdl, encoded) = batch.inputs.logits_with_features(&net)?;
                let [policy, value] = losses((policy, wdl), &batch)?;
                let mut objective = (&policy + (&value * value_loss_weight)?)?;
                if let (Some(head), Some(target)) = (&reply_head, &batch.reply) {
                    objective =
                        (objective + (head.loss(&encoded, target)? * config.reply_weight)?)?;
                }
                let lr =
                    adamw.learning_rate * adamw.schedule.multiplier(optimizer_steps, planned_steps);
                optimizer.step(&objective.backward()?, lr)?;
                optimizer_steps += 1;
                average.update(&parameters, optimizer_steps)?;
                positions_seen += chunk.len();
                let policy = f64::from(policy.to_scalar::<f32>()?);
                let wdl = f64::from(value.to_scalar::<f32>()?);
                totals.add(policy, wdl, chunk.len(), batch.value_count);
                window_totals.add(policy, wdl, chunk.len(), batch.value_count);
                if optimizer_steps.is_multiple_of(METRICS_INTERVAL)
                    || optimizer_steps == planned_steps
                {
                    let [policy, wdl] = window_totals.means();
                    history.push(TrainingSample {
                        policy,
                        wdl,
                        lr,
                        elapsed: training_started.elapsed().as_secs_f64(),
                        step: optimizer_steps,
                        positions: positions_seen,
                        epoch,
                    });
                    window_totals = LossTotals::default();
                }
            }
            let [policy, value] = totals.means();
            if !policy.is_finite() || !value.is_finite() {
                return Err("non-finite training loss".into());
            }
            println!(
                "epoch {epoch}: {}/{} positions; train policy {policy:.4}, WDL {value:.4}, {:.1}s",
                totals.positions,
                rows.len(),
                epoch_started.elapsed().as_secs_f64()
            );
        }
        let phase_completed = phase_steps / steps_per_epoch;
        completed_epochs += phase_completed;
        partial_epoch_positions = (phase_steps % steps_per_epoch) * batch_size;
        phase_metrics.push(json!({
            "phase": index + 1,
            "positions": rows.len(),
            "optimizer_steps": phase_steps,
            "positions_seen": positions_seen - phase_start,
            "epochs": (positions_seen - phase_start) as f64 / rows.len() as f64,
        }));
    }
    device.synchronize()?;
    let evaluation_started = Instant::now();
    let training_seconds = evaluation_started
        .duration_since(training_started)
        .as_secs_f64();

    // Load the trained parameters as constants so the test pass does not build a backward graph.
    let mut tensors: std::collections::HashMap<_, _> = parameters
        .data()
        .lock()
        .unwrap()
        .iter()
        .map(|(name, var)| (name.clone(), var.as_tensor().detach()))
        .collect();
    tensors = average.finish(tensors);
    tensors.retain(|name, _| !name.starts_with("reply."));
    let builder = VarBuilder::from_tensors(tensors.clone(), DType::F32, device);
    let net = Net::new(builder, model)?;
    let mut counts = [1.; 3];
    for row in datasets.iter().flat_map(|data| &data.train) {
        for (count, &probability) in counts.iter_mut().zip(&row.value) {
            *count += f64::from(probability);
        }
    }
    let count_sum = counts.iter().sum::<f64>();
    let mut totals = [0.; 3];
    let mut value_baseline = 0.;
    for chunk in held_out.test.chunks(batch_size) {
        let batch = Batch::new(chunk.iter(), device, 0., &mut hint_rng)?;
        let losses = losses(batch.inputs.logits(&net)?, &batch)?;
        for (total, loss) in totals.iter_mut().zip(losses) {
            *total += f64::from(loss.to_scalar::<f32>()?) * chunk.len() as f64;
        }
        totals[2] += batch.uniform_loss;
        for row in chunk {
            for (count, probability) in counts.iter().zip(row.value) {
                value_baseline -= f64::from(probability) * (count / count_sum).ln();
            }
        }
    }
    let test_count = held_out.test.len() as f64;
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

    println!(
        "epochs: {completed_epochs} complete, {partial_epoch_positions}/{} positions in a partial epoch",
        held_out.train.len()
    );

    let metrics = json!({
        "estimated_generation_seconds": generation_seconds,
        "completed_epochs": completed_epochs,
        "partial_epoch_positions": partial_epoch_positions,
        "optimizer_steps": optimizer_steps,
        "training_phases": phase_metrics,
        "setup_seconds": setup_seconds,
        "training_seconds": training_seconds,
        "evaluation_seconds": evaluation_seconds,
        "dtype": "f32",
        "layer_norm_backward": true,
        "train_games": train_games,
        "train_positions": train_positions,
        "value_train_positions": value_label_count,
        "tablebase_rescoring": rescorer.stats,
        "value_train_games": value_labels.len(),
        "value_labels_per_game_p90": value_p90,
        "test_game_ids": held_out.test_ids,
        "test_positions": held_out.test.len(),
        "test_policy_loss": policy,
        "uniform_policy_loss": uniform,
        "beats_uniform_policy": beats_uniform,
        "test_wdl_loss": value,
        "prior_wdl_loss": value_baseline,
    });
    if matches!(
        model.hints,
        chess_network::hints::HintConfig::Material { .. }
    ) {
        hints.save(&mut tensors)?;
    }
    checkpoint::save(output_dir, tensors, &config, metrics)?;
    let mut csv =
        String::from("loss,loss/policy,loss/wdl,lr,elapsed_time,step,position_seen,epoch\n");
    for TrainingSample {
        policy,
        wdl,
        lr,
        elapsed,
        step,
        positions,
        epoch,
    } in history
    {
        let loss = policy + wdl;
        writeln!(
            csv,
            "{loss},{policy},{wdl},{lr},{elapsed},{step},{positions},{epoch}"
        )?;
    }
    fs::write(output_dir.join("training.csv"), csv)?;
    let target_seconds: f64 = datasets.iter().map(|d| d.target_seconds).sum();
    let timed_seconds = training_started.elapsed().as_secs_f64() + target_seconds;
    let setup_seconds = setup_seconds - target_seconds;
    println!("training target preparation: {target_seconds:.2}s charged");
    println!("saved {}", output_dir.display());
    println!(
        "runtime: {timed_seconds:.2}s timed (training, held-out evaluation, and save); {setup_seconds:.2}s setup excluded; {optimizer_steps} optimizer steps ({training_seconds:.2}s training, {evaluation_seconds:.2}s held-out evaluation)"
    );
    let spent = timed_seconds + GENERATION_BUDGET_WEIGHT * generation_seconds;
    println!(
        "training budget: {spent:.2}s spent ({timed_seconds:.2}s timed runtime + {GENERATION_BUDGET_WEIGHT} × {generation_seconds:.2}s estimated generation)"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use candle_nn::{Optimizer, SGD};
    use chess_config::TomlConfig;
    use chess_network::{INPUTS, SQUARES};

    use super::*;

    #[test]
    fn soft_wdl_loss_matches_weighted_hard_losses_and_gradients() -> Result<()> {
        let device = &Device::Cpu;
        let logits =
            candle_core::Var::from_slice(&[0.3f32, -0.8, 1.2, -0.4, 0.7, 0.2], (2, 3), device)?;
        let original = Tensor::new(&[1u32, 0], device)?;
        let rescored = Tensor::new(&[2u32, 1], device)?;
        for mix in [0f64, 0.25, 1.] {
            let targets = Tensor::new(
                &[
                    [0f32, (1. - mix) as f32, mix as f32],
                    [(1. - mix) as f32, mix as f32, 0.],
                ],
                device,
            )?;
            let soft = soft_value_loss(&logits, &targets)?;
            let hard = ((candle_nn::loss::cross_entropy(&logits, &original)? * (1. - mix))?
                + (candle_nn::loss::cross_entropy(&logits, &rescored)? * mix)?)?;
            assert!((soft.to_scalar::<f32>()? - hard.to_scalar::<f32>()?).abs() < 1e-6);
            let sg = soft.backward()?;
            let hg = hard.backward()?;
            let sg = sg.get(&logits).unwrap().flatten_all()?.to_vec1::<f32>()?;
            let hg = hg.get(&logits).unwrap().flatten_all()?.to_vec1::<f32>()?;
            assert!(sg.iter().zip(hg).all(|(a, b)| (a - b).abs() < 1e-6));
        }
        Ok(())
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
        let hints = Tensor::zeros((1, chess_network::hints::FEATURES), DType::F32, device)?;
        let (policy, _) = net.forward_hinted(&board, &moves, &moves, &promotes, Some(&hints))?;
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

    #[test]
    fn transformer_wdl_loss_reaches_the_shared_trunk() -> Result<()> {
        let mut config = Config::from_repo_file("training.toml")?;
        config.model.layers = 1;
        let device = &Device::Cpu;
        let params = VarMap::new();
        let net = Net::new(
            VarBuilder::from_varmap(&params, DType::F32, device),
            config.model,
        )?;
        let boards = Tensor::randn(0f32, 1., (2, SQUARES, INPUTS), device)?;
        let moves = Tensor::zeros((2, 1), DType::U32, device)?;
        let promotes = Tensor::zeros((2, 1), DType::F32, device)?;
        let hints = Tensor::zeros((2, chess_network::hints::FEATURES), DType::F32, device)?;
        let labels = Tensor::new(&[0_u32, 2], device)?;
        let objective = || -> candle_core::Result<Tensor> {
            let (_, wdl) = net.forward_hinted(&boards, &moves, &moves, &promotes, Some(&hints))?;
            candle_nn::loss::cross_entropy(&wdl, &labels)
        };
        // Zero output initialization needs one update before trunk gradients flow.
        SGD::new(params.all_vars(), 0.1)?.backward_step(&objective()?)?;
        let grads = objective()?.backward()?;
        let params = params.data().lock().unwrap();
        assert!(
            params
                .keys()
                .all(|name| !name.starts_with("piece_count_value"))
        );
        for name in ["input.weight", "transformer_value.output.weight"] {
            let grad = grads.get(params[name].as_tensor()).expect("value gradient");
            assert!(grad.abs()?.sum_all()?.to_scalar::<f32>()? > 0.0, "{name}");
        }
        Ok(())
    }
}
