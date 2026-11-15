use std::{num::NonZeroUsize, path::PathBuf};

use chess_network::ModelConfig;
use serde::Serialize;

use crate::data::Result;

#[chess_config::toml_config]
#[derive(serde::Serialize)]
pub struct Config {
    pub(crate) output_dir: PathBuf,
    pub device: chess_config::DeviceConfig,
    pub(crate) batch_size: NonZeroUsize,
    pub(crate) optimizer: OptimizerConfig,
    pub(crate) seed: u64,
    #[range(min_exclusive = 0.0, max = 1.0)]
    pub(crate) value_sample_rate: f64,
    #[range(min_exclusive = 0.0)]
    pub(crate) value_loss_weight: f64,
    pub(crate) value_rescoring: RescoringConfig,
    pub(crate) curriculum: Vec<CurriculumStep>,
    pub(crate) model: ModelConfig,
}

#[chess_config::toml_config]
#[derive(serde::Serialize)]
pub(crate) struct CurriculumStep {
    pub(crate) dataset: PathBuf,
    pub(crate) num_positions: NonZeroUsize,
    pub(crate) epochs: f64,
}

impl CurriculumStep {
    pub(crate) fn optimizer_steps(&self, positions: usize, batch_size: usize) -> Result<usize> {
        let batches_per_epoch = positions.div_ceil(batch_size);
        let steps = (self.epochs * batches_per_epoch as f64).floor();
        if !(1.0..usize::MAX as f64).contains(&steps) {
            return Err(
                "curriculum epochs must schedule at least one batch and fit in usize".into(),
            );
        }
        Ok(steps as usize)
    }
}

#[derive(serde::Deserialize, serde::Serialize)]
#[serde(tag = "kind", rename_all = "lowercase", deny_unknown_fields)]
pub(crate) enum OptimizerConfig {
    Adamw(AdamWConfig),
}

#[chess_config::toml_config]
#[derive(serde::Serialize)]
pub(crate) struct AdamWConfig {
    #[range(min_exclusive = 0.0)]
    pub(crate) learning_rate: f64,
    #[range(min = 0.0, max_exclusive = 1.0)]
    pub(crate) beta1: f64,
    #[range(min = 0.0, max_exclusive = 1.0)]
    pub(crate) beta2: f64,
    #[range(min_exclusive = 0.0)]
    pub(crate) eps: f64,
    #[range(min = 0.0)]
    pub(crate) weight_decay: f64,
    pub(crate) schedule: Schedule,
}

impl From<&AdamWConfig> for candle_nn::ParamsAdamW {
    fn from(config: &AdamWConfig) -> Self {
        Self {
            lr: config.learning_rate,
            beta1: config.beta1,
            beta2: config.beta2,
            eps: config.eps,
            weight_decay: config.weight_decay,
        }
    }
}

#[chess_config::toml_config]
#[derive(serde::Serialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub(crate) enum Schedule {
    Constant {},
    Cosine {
        #[range(min = 0.0, max_exclusive = 1.0)]
        warmup_fraction: f64,
        #[range(min = 0.0, max = 1.0)]
        min_ratio: f64,
    },
}

impl Schedule {
    pub(crate) fn multiplier(&self, step: usize, steps: usize) -> f64 {
        match *self {
            Self::Constant {} => 1.0,
            Self::Cosine {
                warmup_fraction,
                min_ratio,
            } => {
                let warmup = ((steps as f64 * warmup_fraction).round() as usize).min(steps - 1);
                if step < warmup {
                    (step + 1) as f64 / warmup as f64
                } else {
                    let progress = (step - warmup) as f64 / (steps - warmup - 1).max(1) as f64;
                    min_ratio
                        + (1.0 - min_ratio) * (1.0 + (std::f64::consts::PI * progress).cos()) / 2.0
                }
            }
        }
    }
}

#[chess_config::toml_config]
#[derive(Serialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub(crate) enum RescoringConfig {
    Disabled,
    Chesstb {
        directory: PathBuf,
        #[range(min = 0.0, max = 1.0)]
        policy_mix: f64,
        #[range(min = 0.0, max = 1.0)]
        value_mix: f64,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use chess_config::TomlConfig;
    #[test]
    fn curriculum_epochs_skip_fractional_steps() -> Result<()> {
        let config = Config::from_repo_file("training.toml")?;
        let mut step = config.curriculum.into_iter().next().unwrap();
        step.epochs = 2.0;
        assert_eq!(step.optimizer_steps(129, 128)?, 4);
        step.epochs = 1.25;
        assert_eq!(step.optimizer_steps(129, 128)?, 2);
        step.epochs = 1.75;
        assert_eq!(step.optimizer_steps(129, 128)?, 3);
        step.epochs = 0.8;
        assert_eq!(step.optimizer_steps(640, 128)?, 4);
        step.epochs = 0.0;
        assert!(step.optimizer_steps(129, 128).is_err());
        Ok(())
    }

    #[test]
    fn cosine_schedule_warms_up_and_reaches_its_floor() -> Result<()> {
        let schedule = Schedule::Cosine {
            warmup_fraction: 0.05,
            min_ratio: 0.1,
        };
        assert_eq!(schedule.multiplier(0, 100), 0.2);
        assert_eq!(schedule.multiplier(4, 100), 1.0);
        assert_eq!(schedule.multiplier(5, 100), 1.0);
        assert!(schedule.multiplier(50, 100) < 1.0);
        assert!((schedule.multiplier(99, 100) - 0.1).abs() < 1e-12);
        Ok(())
    }
}
