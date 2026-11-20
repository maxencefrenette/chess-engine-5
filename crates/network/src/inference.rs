use std::{io, path::Path};

#[cfg(feature = "cuda")]
mod cuda;

use candle_core::{DType, Device};
use chess_config::DeviceConfig;
use chess_data::Board;

use crate::{
    INPUTS, Net, SQUARES,
    checkpoint::Checkpoint,
    encode_position,
    policy_inputs::PolicyInputs,
    value::{ValueConfig, score_and_draw},
};

pub struct Prediction {
    pub value: f32,
    pub draw: f32,
    pub priors: Vec<f32>,
}

pub struct InferenceModel {
    net: Net,
    hints: crate::hints::MaterialTable,
    device: Device,
    #[cfg(feature = "cuda")]
    cuda: std::sync::Mutex<cuda::Inference>,
}

impl InferenceModel {
    pub fn load(model_dir: &Path, device: DeviceConfig) -> io::Result<Self> {
        let device = device.try_into().map_err(io::Error::other)?;
        let checkpoint = Checkpoint::load(model_dir, &device)?;
        let hints = checkpoint.hints.clone();
        let net = checkpoint.into_model(&device).map_err(io::Error::other)?;
        Ok(Self {
            hints,
            net,
            device,
            #[cfg(feature = "cuda")]
            cuda: std::sync::Mutex::new(cuda::Inference::default()),
        })
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
        #[cfg(feature = "cuda")]
        if self.device.is_cuda() {
            let (logits, values) = self
                .cuda
                .lock()
                .map_err(|e| io::Error::other(e.to_string()))?
                .evaluate(
                    &self.net,
                    &self.hints,
                    positions,
                    moves.clone(),
                    &self.device,
                )
                .map_err(io::Error::other)?;
            return predictions(moves, logits, values);
        }
        let mut features = Vec::with_capacity(size * SQUARES * INPUTS);
        let mut hints = Vec::with_capacity(size);
        let mut masks = Vec::with_capacity(size);
        for (snapshot, black) in positions {
            let (snapshot, black, mask) = self.net.config.perspective.orient(snapshot, black);
            masks.push(mask);
            hints.push(self.hints.lookup(
                &crate::hints::material_key(&snapshot.board, black),
                self.net.config.hints,
            ));
            encode_position(&mut features, &snapshot, black)
                .ok_or_else(|| io::Error::other("invalid piece code"))?;
        }
        let (policy, wdl) = PolicyInputs::new(
            features,
            moves
                .clone()
                .zip(masks.iter())
                .map(|(row, &mask)| row.map(move |mv| mv ^ mask)),
            &self.device,
        )
        .and_then(|inputs| inputs.with_hints(hints))
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
        predictions(moves, logits, value_logits)
    }
}

fn predictions(
    moves: impl Iterator<Item = impl ExactSizeIterator<Item = u16>>,
    logits: Vec<Vec<f32>>,
    value_logits: Vec<Vec<f32>>,
) -> io::Result<Vec<Prediction>> {
    moves
        .zip(logits)
        .zip(value_logits)
        .map(|((moves, mut priors), wdl)| {
            let (value, draw) = score_and_draw(wdl.try_into().expect("three WDL logits"));
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
            Ok(Prediction {
                value,
                draw,
                priors,
            })
        })
        .collect()
}

#[cfg(all(test, feature = "cuda"))]
mod tests;
