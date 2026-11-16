use std::num::NonZeroUsize;

use candle_core::{DType, Result, Tensor};
use candle_nn::{Embedding, Init, Module, VarBuilder};

mod table;
pub use table::{MaterialKey, MaterialTable, TableBuilder, material_key};

#[chess_config::toml_config]
#[derive(Clone, Copy, Debug, serde::Serialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum HintConfig {
    Disabled {},
    Material {
        min_games: NonZeroUsize,
        injection: Injection,
        #[range(min = 0.0, max_exclusive = 1.0)]
        dropout: f64,
    },
}

impl HintConfig {
    pub fn dropout(self) -> f64 {
        match self {
            Self::Disabled {} => 0.,
            Self::Material { dropout, .. } => dropout,
        }
    }
}

#[derive(Clone, Copy, Debug, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Injection {
    Early,
    EarlyAndValue,
}

pub const FEATURES: usize = 4; // WDL probabilities followed by a trust bucket.
const TRUST_BUCKETS: usize = 33; // Missing, then 8 support bands × 4 concentration bands.

pub(crate) struct HintEmbedding {
    outcomes: Tensor,
    trust: Embedding,
    weight: Embedding,
}

impl HintEmbedding {
    pub(crate) fn new(vb: VarBuilder, width: usize) -> Result<Self> {
        let init = Init::Randn {
            mean: 0.,
            stdev: 0.02,
        };
        Ok(Self {
            outcomes: vb.get_with_hints((3, width), "outcomes", init)?,
            trust: Embedding::new(
                vb.get_with_hints((TRUST_BUCKETS, width), "trust", init)?,
                width,
            ),
            weight: Embedding::new(
                vb.get_with_hints((TRUST_BUCKETS, 1), "trust_weight", Init::Const(1.))?,
                1,
            ),
        })
    }

    pub(crate) fn forward(&self, features: &Tensor) -> Result<Tensor> {
        let probabilities = features
            .narrow(1, 0, 3)?
            .contiguous()?
            .to_dtype(self.outcomes.dtype())?;
        let bucket = features.narrow(1, 3, 1)?.squeeze(1)?.to_dtype(DType::U32)?;
        let hint = probabilities.matmul(&self.outcomes)?;
        let weighted = hint.broadcast_mul(&self.weight.forward(&bucket)?)?;
        // Bucket zero marks missing hints; suppress both embeddings exactly.
        (weighted + self.trust.forward(&bucket)?)?
            .broadcast_mul(
                &bucket
                    .ne(0u32)?
                    .to_dtype(self.outcomes.dtype())?
                    .unsqueeze(1)?,
            )?
            .unsqueeze(1)
    }
}

#[cfg(test)]
mod tests;
