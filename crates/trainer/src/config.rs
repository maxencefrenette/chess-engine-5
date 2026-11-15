use std::{num::NonZeroUsize, path::PathBuf};

use chess_config::{BoundedFloat, Exclusive, Inclusive};
use chess_network::ModelConfig;
use serde::{Deserialize, Serialize};

use crate::data::Result;

#[chess_config::toml_config]
#[derive(serde::Serialize)]
pub struct Config {
    pub(crate) output_dir: PathBuf,
    pub device: chess_config::DeviceConfig,
    pub(crate) batch_size: NonZeroUsize,
    pub(crate) optimizer: OptimizerConfig,
    pub(crate) seed: u64,
    pub(crate) value_sample_rate: BoundedFloat<0, 1, Exclusive, Inclusive>,
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
    pub(crate) learning_rate: f64,
    pub(crate) beta1: BoundedFloat<0, 1, Inclusive, Exclusive>,
    pub(crate) beta2: BoundedFloat<0, 1, Inclusive, Exclusive>,
    pub(crate) eps: f64,
    pub(crate) weight_decay: f64,
    pub(crate) schedule: Schedule,
}

#[derive(serde::Deserialize, serde::Serialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum Schedule {
    Constant {},
    Cosine {
        warmup_fraction: BoundedFloat<0, 1, Inclusive, Exclusive>,
        min_ratio: BoundedFloat<0, 1, Inclusive, Inclusive>,
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
                let warmup =
                    ((steps as f64 * warmup_fraction.get()).round() as usize).min(steps - 1);
                if step < warmup {
                    (step + 1) as f64 / warmup as f64
                } else {
                    let progress = (step - warmup) as f64 / (steps - warmup - 1).max(1) as f64;
                    min_ratio.get()
                        + (1.0 - min_ratio.get()) * (1.0 + (std::f64::consts::PI * progress).cos())
                            / 2.0
                }
            }
        }
    }
}

#[derive(Deserialize, Serialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum RescoringConfig {
    Disabled,
    Chesstb {
        directory: PathBuf,
        policy_mix: BoundedFloat<0, 1, Inclusive, Inclusive>,
        value_mix: BoundedFloat<0, 1, Inclusive, Inclusive>,
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
    fn cosine_schedule_warms_up_and_reaches_its_floor() {
        let schedule = Schedule::Cosine {
            warmup_fraction: BoundedFloat::new(0.05).unwrap(),
            min_ratio: BoundedFloat::new(0.1).unwrap(),
        };
        assert_eq!(schedule.multiplier(0, 100), 0.2);
        assert_eq!(schedule.multiplier(4, 100), 1.0);
        assert_eq!(schedule.multiplier(5, 100), 1.0);
        assert!(schedule.multiplier(50, 100) < 1.0);
        assert!((schedule.multiplier(99, 100) - 0.1).abs() < 1e-12);
    }
}
