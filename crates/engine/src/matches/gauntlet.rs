use std::{collections::HashSet, error::Error, num::NonZeroU64, path::PathBuf};

use chess_config::{PartialConfig, TomlConfig};
use serde::de::DeserializeOwned;

use super::{Mode, ModelEvaluator, Options, SearchSettings, load_openings, nelo_summary, play};

#[chess_config::toml_config]
#[serde(bound(deserialize = "T: DeserializeOwned"))]
struct Plan<T: DeserializeOwned> {
    pairs: NonZeroU64,
    candidates: Vec<T>,
}

#[chess_config::toml_config]
struct ModelCandidate {
    name: String,
    checkpoint: PathBuf,
}

#[chess_config::toml_config]
struct SearchCandidate {
    name: String,
    settings: PartialConfig<SearchSettings>,
}

enum Candidate {
    Model(PathBuf),
    Search(SearchSettings),
}

pub struct Gauntlet {
    pairs: NonZeroU64,
    candidates: Vec<(String, Candidate)>,
}

impl Gauntlet {
    pub fn models(path: &str) -> Result<Self, Box<dyn Error>> {
        let plan = Plan::<ModelCandidate>::from_file(path)?;
        let candidates = plan
            .candidates
            .into_iter()
            .map(|c| (c.name, Candidate::Model(c.checkpoint)))
            .collect();
        Ok(Self {
            pairs: plan.pairs,
            candidates,
        })
    }

    pub fn search(path: &str, baseline: SearchSettings) -> Result<Self, Box<dyn Error>> {
        let plan = Plan::<SearchCandidate>::from_file(path)?;
        let mut candidates = Vec::new();
        for candidate in plan.candidates {
            let settings = candidate
                .settings
                .apply(&baseline)
                .map_err(|error| format!("{}: {error}", candidate.name))?;
            candidates.push((candidate.name, Candidate::Search(settings)));
        }
        Ok(Self {
            pairs: plan.pairs,
            candidates,
        })
    }

    pub fn run(
        self,
        mut options: Options,
        settings: SearchSettings,
        matched_speed: bool,
    ) -> Result<(), Box<dyn Error>> {
        if self.candidates.is_empty() {
            return Err("gauntlet needs at least one candidate".into());
        }
        let mut names = HashSet::new();
        for (name, _) in &self.candidates {
            if name.trim().is_empty() || !names.insert(name) {
                return Err(
                    format!("expected a nonempty, unique candidate name, got {name:?}").into(),
                );
            }
        }
        options.max_pairs = self.pairs.get();
        let openings = load_openings(&options)?;
        let baseline = ModelEvaluator::load(&options.baseline_dir, options.device)?;
        println!(
            "baseline={} seed={}",
            options.baseline_dir.display(),
            options.seed
        );
        let mut results = Vec::new();
        for (name, candidate) in self.candidates {
            println!("Gauntlet candidate {name}");
            let model;
            let (opponent, candidate_settings) = match candidate {
                Candidate::Model(path) => {
                    model = ModelEvaluator::load(&path, options.device)
                        .map_err(|error| format!("{name} ({}): {error}", path.display()))?;
                    (&model, settings)
                }
                Candidate::Search(candidate_settings) => (&baseline, candidate_settings),
            };
            let result = play(
                &options,
                &openings,
                Mode::Fixed,
                matched_speed,
                [&baseline, opponent],
                [settings, candidate_settings],
            )
            .map_err(|error| format!("{name}: {error}"))?;
            results.push((name, result));
        }
        println!("Candidate\tPairs\tW/D/L\tCandidate visits\tnElo (95% CI)");
        for (name, result) in results {
            let [w, d, l] = result.wdl;
            println!(
                "{name}\t{}\t{w}/{d}/{l}\t{}\t{}",
                result.counts.iter().sum::<u64>(),
                result.candidate_visits,
                nelo_summary(result.counts, None)
            );
        }
        Ok(())
    }
}
