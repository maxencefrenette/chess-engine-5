use std::num::NonZeroUsize;

use candle_core::{Result, Tensor};
use candle_nn::{Init, Linear, Module, VarBuilder, linear};

use crate::ModelConfig;

#[derive(Clone, Copy, Debug, serde::Deserialize, serde::Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ValueConfig {
    Pooled { hidden: NonZeroUsize },
}

pub(crate) struct ValueHead {
    hidden: Linear,
    output: Linear,
}

impl ValueHead {
    pub(crate) fn new(vb: VarBuilder, model: ModelConfig) -> Result<Self> {
        let ValueConfig::Pooled { hidden } = model.value;
        let vb = vb.pp("transformer_value");
        let hidden_layer = linear(model.width, hidden.get(), vb.pp("hidden"))?;
        let output = vb.pp("output");
        // Start at uniform WDL probabilities.
        let output = Linear::new(
            output.get_with_hints((3, hidden.get()), "weight", Init::Const(0.))?,
            Some(output.get_with_hints(3, "bias", Init::Const(0.))?),
        );
        Ok(Self {
            hidden: hidden_layer,
            output,
        })
    }

    pub(crate) fn logits(&self, encoded: &Tensor) -> Result<Tensor> {
        self.output
            .forward(&self.hidden.forward(&encoded.mean(1)?)?.relu()?)
    }
}

pub(crate) fn expected_value(logits: [f32; 3]) -> f32 {
    let max = logits.into_iter().fold(f32::NEG_INFINITY, f32::max);
    let wdl = logits.map(|logit| (logit - max).exp());
    let sum = wdl.iter().sum::<f32>();
    wdl[2] / sum - wdl[0] / sum
}
