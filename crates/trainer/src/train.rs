use std::{fs, path::Path, time::Instant};

use candle_core::{DType, Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap, loss, ops::log_softmax};
use chess_network::{BLOCKS, HEADS, INPUTS, Net, WIDTH};
use rand::{SeedableRng, seq::SliceRandom};
use rand_chacha::ChaCha8Rng;
use serde_json::json;

use crate::data::{self, Batch, Result};

const SEED: u64 = 42;
const BATCH_SIZE: usize = 128;
const LEARNING_RATE: f64 = 0.001;

fn losses(net: &Net, batch: &Batch) -> candle_core::Result<(Tensor, Tensor)> {
    let (policy, value) = net.forward(
        &batch.position,
        &batch.move_index,
        &batch.promotion_index,
        &batch.promotes,
        batch.material.as_ref(),
    )?;
    let log_probs = log_softmax(&(&policy + &batch.mask)?, 1)?;
    let policy = (&log_probs * &batch.policy)?.sum(1)?.neg()?.mean_all()?;
    Ok((policy, loss::cross_entropy(&value, &batch.value)?))
}

pub fn run(
    shard: &Path,
    output: &Path,
    epochs: usize,
    device: &Device,
    material_hint: bool,
) -> Result<()> {
    if output.exists() {
        return Err(format!("output already exists: {}", output.display()).into());
    }
    let started = Instant::now();
    let data = data::load(shard, SEED)?;
    println!(
        "train: {} games / {} positions; test: {} games / {} positions",
        data.train_games,
        data.train.len(),
        data.test_ids.len(),
        data.test.len()
    );
    if device.is_cuda() {
        device.set_seed(SEED)?;
    }
    let parameters = VarMap::new();
    let net = Net::new(
        VarBuilder::from_varmap(&parameters, DType::F32, device),
        material_hint,
    )?;
    let mut optimizer = AdamW::new(
        parameters.all_vars(),
        ParamsAdamW {
            lr: LEARNING_RATE,
            ..Default::default()
        },
    )?;
    let mut indices: Vec<_> = (0..data.train.len()).collect();
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    for epoch in 1..=epochs {
        let epoch_started = Instant::now();
        indices.shuffle(&mut rng);
        let mut totals = [0.; 2];
        for chunk in indices.chunks(BATCH_SIZE) {
            let batch = Batch::new(&data.train, chunk, device, material_hint)?;
            let (policy, value) = losses(&net, &batch)?;
            let objective = (&policy + &value)?;
            optimizer.backward_step(&objective)?;
            for (total, loss) in totals.iter_mut().zip([policy, value]) {
                *total += f64::from(loss.to_scalar::<f32>()?) * chunk.len() as f64;
            }
        }
        let [policy, value] = totals.map(|loss| loss / data.train.len() as f64);
        if !policy.is_finite() || !value.is_finite() {
            return Err("non-finite training loss".into());
        }
        println!(
            "epoch {epoch}/{epochs}: train policy {policy:.4}, value {value:.4}, {:.1}s",
            epoch_started.elapsed().as_secs_f64()
        );
    }

    // Load the trained parameters as constants so the test pass does not build a backward graph.
    let tensors = parameters
        .data()
        .lock()
        .unwrap()
        .iter()
        .map(|(name, var)| (name.clone(), var.as_tensor().detach()))
        .collect();
    let net = Net::new(
        VarBuilder::from_tensors(tensors, DType::F32, device),
        material_hint,
    )?;
    let mut counts = [1.; 3];
    for row in &data.train {
        counts[row.value as usize] += 1.;
    }
    let count_sum = counts.iter().sum::<f64>();
    let mut totals = [0.; 3];
    let mut value_baseline = 0.;
    let indices: Vec<_> = (0..data.test.len()).collect();
    for chunk in indices.chunks(BATCH_SIZE) {
        let batch = Batch::new(&data.test, chunk, device, material_hint)?;
        let (policy, value) = losses(&net, &batch)?;
        totals[0] += f64::from(policy.to_scalar::<f32>()?) * chunk.len() as f64;
        totals[1] += f64::from(value.to_scalar::<f32>()?) * chunk.len() as f64;
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

    fs::create_dir(output)?;
    parameters.save(output.join("weights.safetensors"))?;
    let manifest = json!({
        "architecture": "lc0_style_smolgen_v1",
        "position_encoding": "packed_board_planes_v1",
        "move_encoding": "ce5_packed_move_v1",
        "value_order": ["loss", "draw", "win"],
        "width": WIDTH,
        "heads": HEADS,
        "blocks": BLOCKS,
        "input_channels": INPUTS,
        "weights": "weights.safetensors",
        "material_hint": material_hint.then_some("stm_material_squash_8_value_residual_v1"),
        "training": {
            "source": shard,
            "seed": SEED,
            "epochs": epochs,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
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
        }
    });
    fs::write(output.join("model.json"), format!("{manifest:#}\n"))?;
    println!(
        "saved {} in {:.1}s",
        output.display(),
        started.elapsed().as_secs_f64()
    );
    Ok(())
}
