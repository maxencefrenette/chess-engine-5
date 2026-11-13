use std::{
    env, fs,
    num::{NonZeroU16, NonZeroU32, NonZeroUsize},
    path::PathBuf,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use chess_data::{FORMAT_VERSION, Position, Result, write_shard};
use chess_network::inference::Prediction;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde_json::json;

use crate::{
    chess::{Color, Game, Outcome},
    mcts::{Leaf, Search},
    model_evaluator::ModelEvaluator,
    settings::Settings,
};

#[chess_config::toml_config]
pub struct Config {
    pub model: PathBuf,
    pub device: chess_config::DeviceConfig,
    output_dir: PathBuf,
    total_games: NonZeroUsize,
    games_per_shard: NonZeroUsize,
    batch: NonZeroUsize,
    leaves_per_game: NonZeroUsize,
    visits_per_move: NonZeroU16,
    seed: u64,
    temperature: f64,
    playout_cap: PlayoutCap,
}

#[derive(Clone, Copy, serde::Deserialize, serde::Serialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
enum PlayoutCap {
    Disabled {},
    Enabled {
        fast_visits: NonZeroU16,
        full_search_rate: f64,
    },
}

struct PlayedGame {
    positions: Vec<Position>,
    white_value: i8,
}

struct ActiveGame {
    id: usize,
    game: Game,
    search: Search,
    rng: ChaCha8Rng,
    cap_rng: ChaCha8Rng,
    ply: u16,
    full_search: bool,
    positions: Vec<Position>,
}

impl ActiveGame {
    fn new(seed: u64, id: usize, full_search_rate: f64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        rng.set_stream(id as u64);
        let mut cap_rng = ChaCha8Rng::seed_from_u64(seed ^ 0x706c_6179_6f75_7463);
        cap_rng.set_stream(id as u64);
        let full_search = cap_rng.random_bool(full_search_rate);
        Self {
            id,
            game: Game::default(),
            search: Search::new(),
            rng,
            cap_rng,
            ply: 0,
            full_search,
            positions: Vec::new(),
        }
    }

    fn play(&mut self, temperature: f64, full_search_rate: f64) -> Option<Outcome> {
        let played = self.search.choose_move(temperature, &mut self.rng);
        if self.full_search {
            let policy: Vec<_> = self.search.visits().collect();
            let legal_moves: Vec<_> = policy.iter().map(|entry| entry.mv).collect();
            self.positions.push(Position {
                ply: self.ply,
                snapshot: self.game.snapshot(&legal_moves),
                policy: policy
                    .into_iter()
                    .map(|entry| {
                        (
                            entry.mv.packed(),
                            u16::try_from(entry.visits).expect("visits fit in u16"),
                        )
                    })
                    .collect(),
                played_move: played.packed(),
            });
        }
        self.game.play_legal(played);
        self.ply += 1;
        self.full_search = self.cap_rng.random_bool(full_search_rate);
        self.search = Search::new();
        self.game.outcome()
    }
}

fn play_games(
    settings: &Settings,
    config: &Config,
    mut ids: std::ops::Range<usize>,
    mut evaluate: impl FnMut(&[(usize, Leaf)]) -> std::io::Result<Vec<Prediction>>,
) -> Result<Vec<(usize, PlayedGame)>> {
    let (fast_visits, full_search_rate) = match config.playout_cap {
        PlayoutCap::Disabled {} => (config.visits_per_move, 1.0),
        PlayoutCap::Enabled {
            fast_visits,
            full_search_rate,
        } => (fast_visits, full_search_rate),
    };
    let mut completed = Vec::with_capacity(ids.len());
    let mut active: Vec<_> = ids
        .by_ref()
        .take(config.batch.get())
        .map(|id| ActiveGame::new(config.seed, id, full_search_rate))
        .collect();
    while !active.is_empty() {
        let mut pending = Vec::new();
        let mut finished = Vec::new();
        for (i, slot) in active.iter_mut().enumerate() {
            let visits = if slot.full_search {
                config.visits_per_move
            } else {
                fast_visits
            };
            let before = pending.len();
            for _ in 0..config.leaves_per_game.get() {
                let Some(leaf) = slot.search.next_leaf(
                    &slot.game,
                    NonZeroU32::from(visits),
                    settings.exploration,
                ) else {
                    break;
                };
                pending.push((i, leaf));
            }
            if pending.len() == before
                && let Some(outcome) = slot.play(config.temperature, full_search_rate)
            {
                let white_value = match outcome {
                    Outcome::Win(Color::White) => 1,
                    Outcome::Win(Color::Black) => -1,
                    Outcome::Draw => 0,
                };
                finished.push((i, white_value));
            }
        }
        let evaluations = evaluate(&pending)?;
        for ((i, leaf), result) in pending.into_iter().zip(evaluations) {
            active[i]
                .search
                .evaluate(leaf, result.value, &result.priors);
        }
        for (i, white_value) in finished.into_iter().rev() {
            let slot = if let Some(id) = ids.next() {
                std::mem::replace(
                    &mut active[i],
                    ActiveGame::new(config.seed, id, full_search_rate),
                )
            } else {
                active.swap_remove(i)
            };
            completed.push((
                slot.id,
                PlayedGame {
                    positions: slot.positions,
                    white_value,
                },
            ));
        }
    }
    completed.sort_unstable_by_key(|(id, _)| *id);
    Ok(completed)
}

pub fn generate(config: &Config, model: &ModelEvaluator, settings: &Settings) -> Result<PathBuf> {
    if !config.temperature.is_finite() || config.temperature < 0.0 {
        return Err("self-play temperature must be finite and nonnegative".into());
    }
    if let PlayoutCap::Enabled {
        fast_visits,
        full_search_rate,
    } = config.playout_cap
        && (!(full_search_rate > 0.0 && full_search_rate <= 1.0)
            || fast_visits >= config.visits_per_move)
    {
        return Err(
            "playout cap requires a rate in (0, 1] and fewer visits for fast searches".into(),
        );
    }
    let version = config.output_dir.join(format!("v{FORMAT_VERSION}"));
    fs::create_dir_all(&version)?;
    let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let run = version.join(format!("{now}-{}", std::process::id()));
    fs::create_dir(&run)?;
    eprintln!("Writing self-play run to {}", run.display());

    let total_games = config.total_games.get();
    let games_per_shard = config.games_per_shard.get();
    let mut shards = 0;
    let mut positions = 0;
    let mut white_wins = 0;
    let mut black_wins = 0;
    let mut draws = 0;
    for start in (0..total_games).step_by(games_per_shard) {
        let end = start.saturating_add(games_per_shard).min(total_games);
        let started = Instant::now();
        let games = play_games(settings, config, start..end, |pending| {
            model.evaluate(pending)
        })?;
        let elapsed = started.elapsed();
        let shard_positions: usize = games.iter().map(|(_, game)| game.positions.len()).sum();
        if shard_positions == 0 {
            return Err("shard contains no full-search positions".into());
        }
        let generation_ns_per_position =
            u64::try_from(elapsed.as_nanos() / shard_positions as u128)?;
        write_shard(
            &run,
            shards,
            games
                .iter()
                .filter(|(_, game)| !game.positions.is_empty())
                .map(|(id, game)| (*id as u64, game.positions.as_slice(), game.white_value)),
            generation_ns_per_position,
        )?;
        shards += 1;
        positions += shard_positions;
        for (_, game) in games {
            match game.white_value {
                1 => white_wins += 1,
                -1 => black_wins += 1,
                _ => draws += 1,
            }
        }
        eprintln!("{end}/{total_games} games, {positions} positions, {shards} shards");
    }

    let manifest = json!({
        "format_version": FORMAT_VERSION,
        "seed": config.seed,
        "engine_version": env!("CARGO_PKG_VERSION"),
        "engine_revision": env::var("CE5_ENGINE_REVISION").ok(),
        "games": total_games,
        "positions": positions,
        "shards": shards,
        "games_per_shard": games_per_shard,
        "white_wins": white_wins,
        "black_wins": black_wins,
        "draws": draws,
        "start_position": "standard",
        "visits_per_move": config.visits_per_move.get(),
        "playout_cap": config.playout_cap,
        "value": model.value_config(),
        "priors": "smolgen-policy-v1",
        "checkpoint": model.model_dir,
        "device": config.device,
        "game_batch": config.batch.get(),
        "leaves_per_game": config.leaves_per_game.get(),
        "sampling": "chacha8-stream-per-game; temperature applied to root visits",
        "temperature": config.temperature,
        "exploration": settings.exploration,
    });
    let partial = run.join("manifest.json.partial");
    fs::write(&partial, format!("{manifest:#}\n"))?;
    fs::rename(partial, run.join("manifest.json"))?;
    Ok(run)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chess_config::TomlConfig;

    #[test]
    fn refilling_preserves_seeded_games_and_ids() -> Result<()> {
        let mut config = Config::from_repo_file("datagen.toml")?;
        config.visits_per_move = NonZeroU16::new(4).unwrap();
        let settings = Settings::load()?;
        for cap in [
            PlayoutCap::Disabled {},
            PlayoutCap::Enabled {
                fast_visits: NonZeroU16::new(1).unwrap(),
                full_search_rate: 0.5,
            },
        ] {
            config.playout_cap = cap;
            let run = |config: &Config| {
                play_games(&settings, config, 5..12, |pending| {
                    Ok(pending
                        .iter()
                        .map(|(_, leaf)| Prediction {
                            value: 0.0,
                            priors: vec![1.0 / leaf.moves.len() as f32; leaf.moves.len()],
                        })
                        .collect())
                })
            };
            config.batch = NonZeroUsize::new(1).unwrap();
            let expected = run(&config)?;
            for (_, game) in &expected {
                for position in &game.positions {
                    assert_eq!(
                        position
                            .policy
                            .iter()
                            .map(|(_, visits)| u32::from(*visits))
                            .sum::<u32>(),
                        u32::from(config.visits_per_move.get()),
                    );
                }
            }
            assert_eq!(
                expected.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
                (5..12).collect::<Vec<_>>()
            );
            for batch in [2, 3, 10] {
                config.batch = NonZeroUsize::new(batch).unwrap();
                let actual = run(&config)?;
                assert_eq!(actual.len(), expected.len());
                for ((id, game), (expected_id, expected_game)) in actual.iter().zip(&expected) {
                    assert_eq!(id, expected_id);
                    assert_eq!(game.white_value, expected_game.white_value);
                    assert_eq!(game.positions, expected_game.positions);
                }
            }
        }
        Ok(())
    }
}
