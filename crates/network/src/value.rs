use std::num::NonZeroUsize;

use candle_core::{Result, Tensor};
use candle_nn::{Init, Linear, Module, VarBuilder, linear};

use crate::ModelConfig;

#[derive(Clone, Copy, Debug, serde::Deserialize, serde::Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ValueConfig {
    Pooled { hidden: NonZeroUsize },
    Attention { hidden: NonZeroUsize },
}

pub(crate) struct ValueHead {
    query: Option<Linear>,
    hidden: Linear,
    output: Linear,
}

impl ValueHead {
    pub(crate) fn new(vb: VarBuilder, model: ModelConfig) -> Result<Self> {
        let vb = vb.pp("transformer_value");
        let (hidden, query) = match model.value {
            ValueConfig::Pooled { hidden } => (hidden, None),
            ValueConfig::Attention { hidden } => {
                let query = vb.get_with_hints((1, model.width), "query", Init::Const(0.))?;
                (hidden, Some(Linear::new(query, None)))
            }
        };
        let hidden_layer = linear(model.width, hidden.get(), vb.pp("hidden"))?;
        let output = vb.pp("output");
        // Start at uniform WDL probabilities.
        let output = Linear::new(
            output.get_with_hints((3, hidden.get()), "weight", Init::Const(0.))?,
            Some(output.get_with_hints(3, "bias", Init::Const(0.))?),
        );
        Ok(Self {
            query,
            hidden: hidden_layer,
            output,
        })
    }

    pub(crate) fn logits(&self, encoded: &Tensor) -> Result<Tensor> {
        let pooled = match &self.query {
            Some(query) => {
                let weights = candle_nn::ops::softmax(&query.forward(encoded)?, 1)?;
                encoded.broadcast_mul(&weights)?.sum(1)?
            }
            None => encoded.mean(1)?,
        };
        self.output.forward(&self.hidden.forward(&pooled)?.relu()?)
    }
}

pub(crate) fn score_and_draw(logits: [f32; 3]) -> (f32, f32) {
    let max = logits.into_iter().fold(f32::NEG_INFINITY, f32::max);
    let wdl = logits.map(|logit| (logit - max).exp());
    let sum = wdl.iter().sum::<f32>();
    (wdl[2] / sum - wdl[0] / sum, wdl[1] / sum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Var};

    #[test]
    fn zero_query_matches_mean_pooling_and_receives_a_gradient() -> Result<()> {
        let device = &Device::Cpu;
        let identity = Linear::new(Tensor::new(&[[1f32]], device)?, None);
        let mut head = ValueHead {
            query: None,
            hidden: identity.clone(),
            output: identity,
        };
        let squares = Tensor::new(&[[[1f32], [3.]], [[2.], [6.]]], device)?;
        let mean = head.logits(&squares)?;
        let query = Var::new(&[[0f32]], device)?;
        head.query = Some(Linear::new(query.as_tensor().clone(), None));
        let attention = head.logits(&squares)?;
        assert_eq!(attention.to_vec2::<f32>()?, mean.to_vec2::<f32>()?);
        let gradients = attention.sum_all()?.backward()?;
        assert!(gradients.get(&query).unwrap().to_vec2::<f32>()?[0][0] > 0.);
        Ok(())
    }
}
