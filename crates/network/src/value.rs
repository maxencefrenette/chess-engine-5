use std::num::NonZeroUsize;

use candle_core::{Device, Result, Tensor};
use candle_nn::{Init, Linear, Module, VarBuilder, linear};

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
    Hybrid {
        piece_hidden: NonZeroUsize,
        transformer_hidden: NonZeroUsize,
    },
}

impl ValueConfig {
    pub fn uses_piece_counts(self) -> bool {
        !matches!(self, Self::Transformer { .. })
    }
}

struct PieceHead {
    hidden: Option<Linear>,
    output: Linear,
}

impl PieceHead {
    fn new(vb: VarBuilder, hidden: Option<NonZeroUsize>) -> Result<Self> {
        let vb = vb.pp("piece_count_value");
        Ok(Self {
            hidden: hidden
                .map(|width| linear(PIECE_FEATURES, width.get(), vb.pp("hidden")))
                .transpose()?,
            output: linear(hidden.map_or(PIECE_FEATURES, NonZeroUsize::get), 3, vb)?,
        })
    }

    fn logits(&self, features: &Tensor) -> Result<Tensor> {
        match &self.hidden {
            Some(hidden) => self.output.forward(&hidden.forward(features)?.relu()?),
            None => self.output.forward(features),
        }
    }
}

struct TransformerHead {
    square: Option<Linear>,
    hidden: Linear,
    output: Linear,
}

impl TransformerHead {
    fn new(
        vb: VarBuilder,
        width: usize,
        square_channels: Option<NonZeroUsize>,
        hidden: usize,
    ) -> Result<Self> {
        let vb = vb.pp("transformer_value");
        let square = square_channels
            .map(|channels| linear(width, channels.get(), vb.pp("square")))
            .transpose()?;
        let inputs = square_channels.map_or(width, |channels| SQUARES * channels.get());
        let hidden_layer = linear(inputs, hidden, vb.pp("hidden"))?;
        let output = vb.pp("output");
        let output = if square.is_some() {
            linear(hidden, 3, output)?
        } else {
            // Start the pooled correction at zero logits.
            Linear::new(
                output.get_with_hints((3, hidden), "weight", Init::Const(0.))?,
                Some(output.get_with_hints(3, "bias", Init::Const(0.))?),
            )
        };
        Ok(Self {
            square,
            hidden: hidden_layer,
            output,
        })
    }

    fn logits(&self, encoded: &Tensor) -> Result<Tensor> {
        let board = match &self.square {
            Some(square) => square.forward(encoded)?.relu()?.flatten_from(1)?,
            None => encoded.mean(1)?,
        };
        self.output.forward(&self.hidden.forward(&board)?.relu()?)
    }
}

pub(crate) struct ValueHead {
    piece: Option<PieceHead>,
    transformer: Option<TransformerHead>,
}

impl ValueHead {
    pub(crate) fn new(vb: VarBuilder, model: ModelConfig) -> Result<Self> {
        let (piece, transformer) = match model.value {
            ValueConfig::PieceLinear {} => (Some(PieceHead::new(vb, None)?), None),
            ValueConfig::PieceMlp { hidden } => (Some(PieceHead::new(vb, Some(hidden))?), None),
            ValueConfig::Transformer {
                square_channels,
                hidden,
            } => (
                None,
                Some(TransformerHead::new(
                    vb,
                    model.width,
                    Some(square_channels),
                    hidden.get(),
                )?),
            ),
            ValueConfig::Hybrid {
                piece_hidden,
                transformer_hidden,
            } => (
                Some(PieceHead::new(vb.clone(), Some(piece_hidden))?),
                Some(TransformerHead::new(
                    vb,
                    model.width,
                    None,
                    transformer_hidden.get(),
                )?),
            ),
        };
        Ok(Self { piece, transformer })
    }

    pub(crate) fn logits(
        &self,
        encoded: &Tensor,
        pieces: Option<&Tensor>,
    ) -> Result<Option<Tensor>> {
        let piece = self
            .piece
            .as_ref()
            .zip(pieces)
            .map(|(head, features)| head.logits(features))
            .transpose()?;
        let transformer = self
            .transformer
            .as_ref()
            .map(|head| head.logits(encoded))
            .transpose()?;
        match (piece, transformer) {
            (Some(piece), Some(transformer)) => Ok(Some((&piece + &transformer)?)),
            (piece, transformer) => Ok(piece.or(transformer)),
        }
    }

    pub(crate) fn cpu_piece_value(&self) -> Result<Option<CpuPieceValue>> {
        self.piece
            .as_ref()
            .map(|head| {
                Ok(CpuPieceValue {
                    hidden: head.hidden.as_ref().map(CpuLinear::load).transpose()?,
                    output: CpuLinear::load(&head.output)?,
                })
            })
            .transpose()
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
    pub(crate) fn logits(&self, board: &[u8; 32], black: bool) -> [f32; 3] {
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
        logits
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
