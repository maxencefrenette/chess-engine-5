use std::collections::HashMap;

use candle_core::{Result, Tensor};
use candle_nn::VarMap;

use crate::config::WeightAveraging;

/// Average post-update weights every ten steps within the final fraction of training.
pub(crate) struct TailAverage {
    start: usize,
    steps: usize,
    samples: usize,
    weights: HashMap<String, Tensor>,
}

impl TailAverage {
    pub(crate) fn new(config: &WeightAveraging, steps: usize) -> Self {
        let start = match config {
            WeightAveraging::Disabled {} => steps,
            WeightAveraging::Tail { fraction } => {
                steps - ((steps as f64 * fraction).ceil() as usize).min(steps)
            }
        };
        Self {
            start,
            steps,
            samples: 0,
            weights: HashMap::new(),
        }
    }

    pub(crate) fn update(&mut self, parameters: &VarMap, step: usize) -> Result<()> {
        if step <= self.start || (!step.is_multiple_of(10) && step != self.steps) {
            return Ok(());
        }
        self.samples += 1;
        for (name, var) in parameters.data().lock().unwrap().iter() {
            let value = var.as_tensor().detach();
            let mean = if let Some(previous) = self.weights.get(name) {
                (previous + ((&value - previous)? / self.samples as f64)?)?
            } else {
                value.copy()?
            };
            self.weights.insert(name.clone(), mean);
        }
        Ok(())
    }

    pub(crate) fn finish(self, final_weights: HashMap<String, Tensor>) -> HashMap<String, Tensor> {
        if self.samples == 0 {
            final_weights
        } else {
            self.weights
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Var};

    #[test]
    fn averages_only_tail_snapshots_without_modifying_live_weights() -> Result<()> {
        let parameters = VarMap::new();
        let weight = Var::new(&[0f32], &Device::Cpu)?;
        parameters
            .data()
            .lock()
            .unwrap()
            .insert("weight".into(), weight.clone());
        let mut average = TailAverage::new(&WeightAveraging::Tail { fraction: 0.5 }, 25);
        for step in 1..=25 {
            weight.set(&Tensor::new(&[step as f32], &Device::Cpu)?)?;
            average.update(&parameters, step)?;
        }
        assert_eq!(weight.to_vec1::<f32>()?, vec![25.]);
        assert_eq!(
            average.finish(HashMap::new())["weight"].to_vec1::<f32>()?,
            vec![22.5]
        );
        Ok(())
    }

    #[test]
    fn disabled_averaging_keeps_final_weights() -> Result<()> {
        let mut average = TailAverage::new(&WeightAveraging::Disabled {}, 10);
        average.update(&VarMap::new(), 10)?;
        let final_weights = HashMap::from([("weight".into(), Tensor::new(&[7f32], &Device::Cpu)?)]);
        assert_eq!(
            average.finish(final_weights)["weight"].to_vec1::<f32>()?,
            vec![7.]
        );
        Ok(())
    }
}
