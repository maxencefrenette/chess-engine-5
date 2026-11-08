use std::{fs, io, path::Path};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use chess_network::{BLOCKS, HEADS, INPUTS, Net, SQUARES, WIDTH, encode_position, move_indices};
use serde_json::{Value, json};

use crate::{
    chess::Color,
    matches::{Candidate, Evaluation, Options, run_match},
    mcts::Leaf,
};

struct Hybrid {
    policy: Net,
    value: [[f32; 10]; 3],
    bias: [f32; 3],
    device: Device,
}

impl Hybrid {
    fn load(model_dir: &Path, value_path: &Path, device: Device) -> io::Result<Self> {
        let model: Value = serde_json::from_slice(&fs::read(model_dir.join("model.json"))?)?;
        for (key, expected) in [
            ("architecture", json!("lc0_style_smolgen_v1")),
            ("position_encoding", json!("packed_board_planes_v1")),
            ("move_encoding", json!("ce5_packed_move_v1")),
            ("width", json!(WIDTH)),
            ("heads", json!(HEADS)),
            ("blocks", json!(BLOCKS)),
            ("input_channels", json!(INPUTS)),
        ] {
            if model[key] != expected {
                return Err(io::Error::other(format!(
                    "unsupported model {key}: {}",
                    model[key]
                )));
            }
        }
        let weights =
            candle_core::safetensors::load(model_dir.join("weights.safetensors"), &device)
                .map_err(io::Error::other)?;
        let policy = Net::new(
            VarBuilder::from_tensors(weights, DType::F32, &device),
            !model["material_hint"].is_null(),
        )
        .map_err(io::Error::other)?;

        let coefficients: Value = serde_json::from_slice(&fs::read(value_path)?)?;
        if coefficients["class_order"] != json!(["loss", "draw", "win"])
            || coefficients["piece_counts"]["features"]
                != "side-to-move P,N,B,R,Q counts followed by opponent counts, divided by 8,2,2,2,1 for each side"
        {
            return Err(io::Error::other("unsupported piece-count value format"));
        }
        let value = serde_json::from_value(coefficients["piece_counts"]["weights"].clone())?;
        let bias = serde_json::from_value(coefficients["piece_counts"]["bias"].clone())?;
        Ok(Self {
            policy,
            value,
            bias,
            device,
        })
    }

    fn logits(
        &self,
        batch: &[(usize, Leaf)],
        values: &mut Vec<f32>,
    ) -> candle_core::Result<Vec<Vec<f32>>> {
        let size = batch.len();
        let width = batch
            .iter()
            .map(|(_, leaf)| leaf.moves.len())
            .max()
            .unwrap();
        let mut positions = Vec::with_capacity(size * SQUARES * INPUTS);
        let mut moves = vec![0; size * width];
        let mut promotions = moves.clone();
        let mut promotes = vec![0.; size * width];
        for (i, (_, leaf)) in batch.iter().enumerate() {
            let snapshot = leaf.game.snapshot(&leaf.moves);
            let black = leaf.game.turn() == Color::Black;
            encode_position(
                &mut positions,
                &snapshot.board,
                black,
                snapshot.castling,
                snapshot.ep_square,
                snapshot.halfmove_clock,
                snapshot.repetitions,
            )
            .expect("engine snapshot has valid pieces");
            values.push(self.value_for(&snapshot.board, black));
            for (j, mv) in leaf.moves.iter().enumerate() {
                let k = i * width + j;
                (moves[k], promotions[k], promotes[k]) = move_indices(mv.packed());
            }
        }
        let shape = (size, width);
        self.policy
            .policy(
                &Tensor::from_vec(positions, (size, SQUARES, INPUTS), &self.device)?,
                &Tensor::from_vec(moves, shape, &self.device)?,
                &Tensor::from_vec(promotions, shape, &self.device)?,
                &Tensor::from_vec(promotes, shape, &self.device)?,
            )?
            .to_vec2()
    }

    fn value_for(&self, board: &[u8; 32], black: bool) -> f32 {
        let mut pieces = [0.; 10];
        for byte in board {
            for piece in [byte & 15, byte >> 4] {
                let side = usize::from(piece > 6);
                let kind = usize::from(piece).saturating_sub(side * 6);
                if (1..=5).contains(&kind) {
                    pieces[(side ^ usize::from(black)) * 5 + kind - 1] += 1.;
                }
            }
        }
        let normalized = pieces
            .into_iter()
            .zip([8., 2., 2., 2., 1., 8., 2., 2., 2., 1.])
            .map(|(n, scale)| n / scale);
        let mut logits = self.bias;
        for (logit, weights) in logits.iter_mut().zip(&self.value) {
            *logit += weights
                .iter()
                .zip(normalized.clone())
                .map(|(w, x)| w * x)
                .sum::<f32>();
        }
        normalize(&mut logits);
        logits[2] - logits[0]
    }
}

fn normalize(logits: &mut [f32]) {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    for logit in logits.iter_mut() {
        *logit = (*logit - max).exp();
    }
    let sum = logits.iter().sum::<f32>();
    for logit in logits {
        *logit /= sum;
    }
}

impl Candidate for Hybrid {
    fn evaluate(&self, batch: &[(usize, Leaf)]) -> io::Result<Vec<Evaluation>> {
        let mut values = Vec::with_capacity(batch.len());
        let logits = self.logits(batch, &mut values).map_err(io::Error::other)?;
        batch
            .iter()
            .zip(logits)
            .zip(values)
            .map(|(((_, leaf), mut priors), value)| {
                priors.truncate(leaf.moves.len());
                if !value.is_finite() || priors.iter().any(|logit| !logit.is_finite()) {
                    return Err(io::Error::other("non-finite hybrid prediction"));
                }
                normalize(&mut priors);
                Ok(Evaluation {
                    value,
                    priors: Some(priors),
                })
            })
            .collect()
    }
}

pub fn run(options: Options, model: &Path, value: &Path, device: &str) -> io::Result<()> {
    let backend = match device {
        "cpu" => Device::Cpu,
        "cuda" if cfg!(feature = "cuda") => Device::new_cuda(0).map_err(io::Error::other)?,
        "cuda" => return Err(io::Error::other("build chess-engine with --features cuda")),
        _ => return Err(io::Error::other("device must be cpu or cuda")),
    };
    let hybrid = Hybrid::load(model, value, backend)?;
    println!(
        "policy={} value={} device={device}",
        model.display(),
        value.display()
    );
    run_match(options, Some(&hybrid))
}
