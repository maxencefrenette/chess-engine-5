use candle_core::{Device, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear};

pub const PIECE_FEATURES: usize = 10;
pub(crate) const PREFIX: &str = "piece_count_value";

pub struct PieceValue(Linear);

impl PieceValue {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        Ok(Self(linear(PIECE_FEATURES, 3, vb.pp(PREFIX))?))
    }

    pub fn logits(&self, features: &Tensor) -> Result<Tensor> {
        self.0.forward(features)
    }

    pub(crate) fn into_cpu(self) -> Result<CpuPieceValue> {
        let weights = self.0.weight().flatten_all()?.to_vec1::<f32>()?;
        let bias = self
            .0
            .bias()
            .expect("linear layer has a bias")
            .to_vec1::<f32>()?;
        let mut weight = [[0.; PIECE_FEATURES]; 3];
        for (row, source) in weight.iter_mut().zip(weights.chunks_exact(PIECE_FEATURES)) {
            row.copy_from_slice(source);
        }
        Ok(CpuPieceValue {
            weight,
            bias: bias.try_into().expect("linear layer has three outputs"),
        })
    }
}

pub struct CpuPieceValue {
    weight: [[f32; PIECE_FEATURES]; 3],
    bias: [f32; 3],
}

impl CpuPieceValue {
    pub fn evaluate(&self, board: &[u8; 32], black: bool) -> f32 {
        let features = piece_features(board, black);
        let mut logits = self.bias;
        for (logit, weights) in logits.iter_mut().zip(&self.weight) {
            *logit += weights
                .iter()
                .zip(features)
                .map(|(w, x)| w * x)
                .sum::<f32>();
        }
        let max = logits.into_iter().fold(f32::NEG_INFINITY, f32::max);
        let wdl = logits.map(|logit| (logit - max).exp());
        let sum = wdl.iter().sum::<f32>();
        wdl[2] / sum - wdl[0] / sum
    }
}

pub fn piece_features(board: &[u8; 32], black: bool) -> [f32; PIECE_FEATURES] {
    let mut pieces = [0.; PIECE_FEATURES];
    for byte in board {
        for piece in [byte & 15, byte >> 4] {
            let side = usize::from(piece > 6);
            let kind = usize::from(piece).saturating_sub(side * 6);
            if (1..=5).contains(&kind) {
                pieces[(side ^ usize::from(black)) * 5 + kind - 1] += 1.;
            }
        }
    }
    for (count, scale) in pieces
        .iter_mut()
        .zip([8., 2., 2., 2., 1., 8., 2., 2., 2., 1.])
    {
        *count /= scale;
    }
    pieces
}

pub fn feature_tensor(features: &[[f32; PIECE_FEATURES]], device: &Device) -> Result<Tensor> {
    Tensor::from_slice(
        features.as_flattened(),
        (features.len(), PIECE_FEATURES),
        device,
    )
}
