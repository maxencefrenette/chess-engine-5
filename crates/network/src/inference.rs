use std::{io, path::Path};

use candle_core::{DType, Device};
use chess_config::DeviceConfig;
use chess_data::Board;

use crate::{
    INPUTS, Net, SQUARES,
    checkpoint::Checkpoint,
    encode_position,
    policy_inputs::PolicyInputs,
    value::{ValueConfig, expected_value},
};

pub struct Prediction {
    pub value: f32,
    pub priors: Vec<f32>,
}

pub struct InferenceModel {
    net: Net,
    device: Device,
}

impl InferenceModel {
    pub fn load(model_dir: &Path, device: DeviceConfig) -> io::Result<Self> {
        let device = device.try_into().map_err(io::Error::other)?;
        let checkpoint = Checkpoint::load(model_dir, &device)?;
        let net = checkpoint.into_model(&device).map_err(io::Error::other)?;
        Ok(Self { net, device })
    }

    pub fn value_config(&self) -> ValueConfig {
        self.net.config.value
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
        }
        let (policy, wdl) = PolicyInputs::new(features, moves.clone(), &self.device)
            .and_then(|inputs| inputs.logits(&self.net))
            .map_err(io::Error::other)?;
        let logits = policy
            .to_dtype(DType::F32)
            .and_then(|x| x.to_vec2::<f32>())
            .map_err(io::Error::other)?;
        let value_logits = wdl
            .to_dtype(DType::F32)
            .and_then(|x| x.to_vec2::<f32>())
            .map_err(io::Error::other)?;
        moves
            .zip(logits)
            .zip(value_logits)
            .map(|((moves, mut priors), wdl)| {
                let value = expected_value(wdl.try_into().expect("three WDL logits"));
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
