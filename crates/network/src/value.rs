use std::num::NonZeroUsize;

use candle_core::{Device, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear};

use crate::{ModelConfig, SQUARES};

pub const PIECE_FEATURES: usize = 10;

#[derive(Clone, Copy, Debug, serde::Deserialize, serde::Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ValueConfig {
    PieceLinear {},
    PieceMlp {
        hidden: NonZeroUsize,
    },
    Transformer {
        square_channels: NonZeroUsize,
        hidden: NonZeroUsize,
    },
}

impl ValueConfig {
    pub fn uses_piece_counts(self) -> bool {
        !matches!(self, Self::Transformer { .. })
    }
}

pub(crate) enum ValueHead {
    Piece {
        hidden: Option<Linear>,
        output: Linear,
    },
    Transformer {
        square: Linear,
        hidden: Linear,
        output: Linear,
    },
}

impl ValueHead {
    pub(crate) fn new(vb: VarBuilder, model: ModelConfig) -> Result<Self> {
        Ok(match model.value {
            ValueConfig::PieceLinear {} => Self::Piece {
                hidden: None,
                output: linear(PIECE_FEATURES, 3, vb.pp("piece_count_value"))?,
            },
            ValueConfig::PieceMlp { hidden } => {
                let vb = vb.pp("piece_count_value");
                Self::Piece {
                    hidden: Some(linear(PIECE_FEATURES, hidden.get(), vb.pp("hidden"))?),
                    output: linear(hidden.get(), 3, vb)?,
                }
            }
            ValueConfig::Transformer {
                square_channels,
                hidden,
            } => {
                let vb = vb.pp("transformer_value");
                Self::Transformer {
                    square: linear(model.width, square_channels.get(), vb.pp("square"))?,
                    hidden: linear(
                        SQUARES * square_channels.get(),
                        hidden.get(),
                        vb.pp("hidden"),
                    )?,
                    output: linear(hidden.get(), 3, vb.pp("output"))?,
                }
            }
        })
    }

    pub(crate) fn logits(
        &self,
        encoded: &Tensor,
        pieces: Option<&Tensor>,
    ) -> Result<Option<Tensor>> {
        match self {
            Self::Piece { hidden, output } => pieces
                .map(|features| match hidden {
                    Some(hidden) => output.forward(&hidden.forward(features)?.relu()?),
                    None => output.forward(features),
                })
                .transpose(),
            Self::Transformer {
                square,
                hidden,
                output,
            } => {
                let board = square.forward(encoded)?.relu()?.flatten_from(1)?;
                Ok(Some(output.forward(&hidden.forward(&board)?.relu()?)?))
            }
        }
    }

    pub(crate) fn cpu_piece_value(&self) -> Result<Option<CpuPieceValue>> {
        match self {
            Self::Piece { hidden, output } => Ok(Some(CpuPieceValue {
                hidden: hidden.as_ref().map(CpuLinear::load).transpose()?,
                output: CpuLinear::load(output)?,
            })),
            Self::Transformer { .. } => Ok(None),
        }
    }
}

struct CpuLinear {
    weights: Vec<Vec<f32>>,
    bias: Vec<f32>,
}

impl CpuLinear {
    fn load(linear: &Linear) -> Result<Self> {
        let cpu = &Device::Cpu;
        Ok(Self {
            weights: linear.weight().to_device(cpu)?.to_vec2()?,
            bias: linear
                .bias()
                .expect("linear layer has a bias")
                .to_device(cpu)?
                .to_vec1()?,
        })
    }

    fn forward(&self, features: &[f32], outputs: &mut [f32]) {
        for ((out, weights), bias) in outputs.iter_mut().zip(&self.weights).zip(&self.bias) {
            *out = bias
                + weights
                    .iter()
                    .zip(features)
                    .map(|(w, x)| w * x)
                    .sum::<f32>();
        }
    }
}

pub(crate) struct CpuPieceValue {
    hidden: Option<CpuLinear>,
    output: CpuLinear,
}

impl CpuPieceValue {
    pub(crate) fn evaluate(&self, board: &[u8; 32], black: bool) -> f32 {
        let pieces = piece_features(board, black);
        let mut logits = [0.; 3];
        if let Some(hidden) = &self.hidden {
            let mut features = vec![0.; hidden.bias.len()];
            hidden.forward(&pieces, &mut features);
            for feature in &mut features {
                *feature = feature.max(0.);
            }
            self.output.forward(&features, &mut logits);
        } else {
            self.output.forward(&pieces, &mut logits);
        }
        expected_value(logits)
    }
}

pub(crate) fn expected_value(logits: [f32; 3]) -> f32 {
    let max = logits.into_iter().fold(f32::NEG_INFINITY, f32::max);
    let wdl = logits.map(|logit| (logit - max).exp());
    let sum = wdl.iter().sum::<f32>();
    wdl[2] / sum - wdl[0] / sum
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
