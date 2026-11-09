use std::{
    io,
    path::{Path, PathBuf},
};

use candle_core::{Device, Tensor};
use chess_network::{
    INPUTS, Net, SQUARES,
    checkpoint::{Checkpoint, PieceValue},
    encode_position, move_indices,
};

use crate::{
    chess::{Color, Game},
    mcts::{Leaf, MoveVisits, Search},
};

pub(crate) struct Evaluation {
    pub value: f32,
    pub priors: Vec<f32>,
}

pub struct Hybrid {
    policy: Net,
    value: PieceValue,
    device: Device,
    pub(crate) model_dir: PathBuf,
}

impl Hybrid {
    pub fn load(model_dir: &Path, device: &str) -> io::Result<Self> {
        let device = match device {
            "cpu" => Device::Cpu,
            "cuda" if cfg!(feature = "cuda") => Device::new_cuda(0).map_err(io::Error::other)?,
            "cuda" => return Err(io::Error::other("build chess-engine with --features cuda")),
            _ => return Err(io::Error::other("device must be cpu or cuda")),
        };
        let checkpoint = Checkpoint::load(model_dir, &device)?;
        let value = checkpoint.value()?;
        let policy = checkpoint.into_policy(&device).map_err(io::Error::other)?;
        Ok(Self {
            policy,
            value,
            device,
            model_dir: model_dir.into(),
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
        let mut logits = self.value.bias;
        for (logit, weights) in logits.iter_mut().zip(&self.value.weight) {
            *logit += weights
                .iter()
                .zip(normalized.clone())
                .map(|(w, x)| w * x)
                .sum::<f32>();
        }
        normalize(&mut logits);
        logits[2] - logits[0]
    }
    pub fn search(&self, game: &Game, visits: std::num::NonZeroU32) -> io::Result<Vec<MoveVisits>> {
        if game.outcome().is_some() {
            return Ok(Vec::new());
        }
        let mut search = Search::new();
        while let Some(leaf) = search.next_leaf(game, visits) {
            let batch = [(0, leaf)];
            let result = self.evaluate(&batch)?.pop().expect("one prediction");
            let [(_, leaf)] = batch;
            search.evaluate(leaf.moves, result.value, &result.priors);
        }
        Ok(search.visits().collect())
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

impl Hybrid {
    pub(crate) fn evaluate(&self, batch: &[(usize, Leaf)]) -> io::Result<Vec<Evaluation>> {
        if batch.is_empty() {
            return Ok(Vec::new());
        }
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
                Ok(Evaluation { value, priors })
            })
            .collect()
    }
}
