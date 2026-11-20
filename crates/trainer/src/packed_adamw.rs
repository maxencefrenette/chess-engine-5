//! Apply AdamW to a flat buffer so each elementwise operation launches once per step.
use candle_core::{DType, Result, Tensor, Var, backprop::GradStore};
use candle_nn::ParamsAdamW;

pub struct PackedAdamW {
    variables: Vec<Var>,
    weights: Tensor,
    first: Tensor,
    second: Tensor,
    params: ParamsAdamW,
    step: i32,
}

impl PackedAdamW {
    pub fn new(variables: Vec<Var>, params: ParamsAdamW) -> Result<Self> {
        let flat = variables
            .iter()
            .map(|v| v.detach().flatten_all())
            .collect::<Result<Vec<_>>>()?;
        let weights = Tensor::cat(&flat, 0)?;
        Ok(Self {
            first: weights.zeros_like()?,
            second: weights.zeros_like()?,
            variables,
            weights,
            params,
            step: 0,
        })
    }

    pub fn step(&mut self, gradients: &GradStore, lr: f64) -> Result<()> {
        self.step += 1;
        let flat = self
            .variables
            .iter()
            .map(|v| match gradients.get(v) {
                Some(g) => g.detach().flatten_all(),
                None => Tensor::zeros(v.elem_count(), v.dtype(), v.device()),
            })
            .collect::<Result<Vec<_>>>()?;
        let g = Tensor::cat(&flat, 0)?;
        let ParamsAdamW {
            beta1,
            beta2,
            eps,
            weight_decay,
            ..
        } = self.params;
        let m = ((&self.first * beta1)? + (&g * (1. - beta1))?)?;
        let v = ((&self.second * beta2)? + (g.sqr()? * (1. - beta2))?)?;
        let m_hat = (&m * (1. - beta1.powi(self.step)).recip())?;
        let v_hat = (&v * (1. - beta2.powi(self.step)).recip())?;
        let delta = ((m_hat / (v_hat.sqrt()? + eps)?)? * lr)?;
        let weights = ((&self.weights * (1. - lr * weight_decay))? - delta)?;

        if self.variables.iter().any(|v| gradients.get(v).is_none()) {
            // A head can receive no labels in a batch. Candle skips its decay and moment updates.
            let masks = self
                .variables
                .iter()
                .map(|v| {
                    let n = v.elem_count();
                    if gradients.get(v).is_some() {
                        Tensor::ones(n, DType::U8, v.device())
                    } else {
                        Tensor::zeros(n, DType::U8, v.device())
                    }
                })
                .collect::<Result<Vec<_>>>()?;
            let mask = Tensor::cat(&masks, 0)?;
            self.weights = mask.where_cond(&weights, &self.weights)?;
            self.first = mask.where_cond(&m, &self.first)?;
            self.second = mask.where_cond(&v, &self.second)?;
        } else {
            (self.weights, self.first, self.second) = (weights, m, v);
        }

        let mut offset = 0;
        for var in &self.variables {
            let n = var.elem_count();
            var.set(&self.weights.narrow(0, offset, n)?.reshape(var.shape())?)?;
            offset += n;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::{AdamW, Optimizer};

    #[test]
    fn matches_candle_when_heads_skip_steps_and_lr_changes() -> Result<()> {
        let init = || -> Result<Vec<Var>> {
            Ok(vec![
                Var::new(&[[1_f32, -2.], [0.5, 3.]], &Device::Cpu)?,
                Var::new(0.75_f32, &Device::Cpu)?,
            ])
        };
        let vars = init()?;
        let refs = init()?;
        let mut packed = PackedAdamW::new(vars.clone(), ParamsAdamW::default())?;
        let mut reference = AdamW::new(refs.clone(), ParamsAdamW::default())?;
        for (step, active) in [
            [true, true],
            [true, false],
            [false, false],
            [false, true],
            [true, true],
        ]
        .into_iter()
        .enumerate()
        {
            let lr = 0.001 / (1 + step) as f64;
            let gradients = |weights: &[Var]| -> Result<_> {
                let mut grads = GradStore::default();
                for (var, active) in weights.iter().zip(active) {
                    if active {
                        grads.insert(var, var.detach().affine(0.25, 0.125)?);
                    }
                }
                Ok(grads)
            };
            packed.step(&gradients(&vars)?, lr)?;
            reference.set_learning_rate(lr);
            reference.step(&gradients(&refs)?)?;
            for (var, expected) in vars.iter().zip(&refs) {
                assert_eq!(
                    var.flatten_all()?.to_vec1::<f32>()?,
                    expected.flatten_all()?.to_vec1::<f32>()?
                );
            }
        }
        Ok(())
    }
}
