use std::{io, path::Path};

use candle_core::Device;
use chess_config::DeviceConfig;
use chess_data::Board;

use crate::{
    INPUTS, Net, SQUARES, checkpoint::Checkpoint, encode_position, piece_value::CpuPieceValue,
    policy_inputs::PolicyInputs,
};

pub struct Prediction {
    pub value: f32,
    pub priors: Vec<f32>,
}

pub struct InferenceModel {
    policy: Net,
    value: CpuPieceValue,
    device: Device,
}

impl InferenceModel {
    pub fn load(model_dir: &Path, device: DeviceConfig) -> io::Result<Self> {
        let device = device.try_into().map_err(io::Error::other)?;
        let checkpoint = Checkpoint::load(model_dir, &device)?;
        let value = checkpoint.value().map_err(io::Error::other)?;
        let policy = checkpoint.into_policy(&device).map_err(io::Error::other)?;
        Ok(Self {
            policy,
            value,
            device,
        })
    }

    pub fn evaluate(
        &self,
        positions: impl ExactSizeIterator<Item = (Board, bool)>,
        moves: impl ExactSizeIterator<Item = impl ExactSizeIterator<Item = u16>> + Clone,
    ) -> io::Result<Vec<Prediction>> {
        let size = positions.len();
        if size != moves.len() {
            return Err(io::Error::other(
                "expected a legal move list for each position",
            ));
        }
        if size == 0 {
            return Ok(Vec::new());
        }
        let mut features = Vec::with_capacity(size * SQUARES * INPUTS);
        let mut values = Vec::with_capacity(size);
        for (snapshot, black) in positions {
            encode_position(
                &mut features,
                &snapshot.board,
                black,
                snapshot.castling,
                snapshot.ep_square,
                snapshot.halfmove_clock,
                snapshot.repetitions,
            )
            .ok_or_else(|| io::Error::other("invalid piece code"))?;
            values.push(self.value.evaluate(&snapshot.board, black));
        }
        let logits = PolicyInputs::new(features, moves.clone(), &self.device)
            .and_then(|inputs| inputs.logits(&self.policy)?.to_vec2::<f32>())
            .map_err(io::Error::other)?;
        moves
            .zip(logits)
            .zip(values)
            .map(|((moves, mut priors), value)| {
                priors.truncate(moves.len());
                if priors.is_empty()
                    || !value.is_finite()
                    || priors.iter().any(|logit| !logit.is_finite())
                {
                    return Err(io::Error::other("empty or non-finite model prediction"));
                }
                let max = priors.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                for logit in &mut priors {
                    *logit = (*logit - max).exp();
                }
                let sum = priors.iter().sum::<f32>();
                for prior in &mut priors {
                    *prior /= sum;
                }
                Ok(Prediction { value, priors })
            })
            .collect()
    }
}
