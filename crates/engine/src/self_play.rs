use std::{
    env, fs,
    num::{NonZeroU16, NonZeroU32, NonZeroUsize},
    path::PathBuf,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use chess_data::{FORMAT_VERSION, Position, Result, write_shard};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde_json::json;

use crate::{
    chess::{Color, Game, Outcome},
    hybrid::Hybrid,
    mcts::Search,
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
    visits_per_move: NonZeroU16,
    seed: u64,
    temperature: f64,
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
    positions: Vec<Position>,
}

impl ActiveGame {
    fn new(seed: u64, id: usize) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        rng.set_stream(id as u64);
        Self {
            id,
            game: Game::default(),
            search: Search::new(),
            rng,
            positions: Vec::new(),
        }
    }

    fn play(&mut self, temperature: f64) -> Option<Outcome> {
        let policy: Vec<_> = self.search.visits().collect();
        let played = self.search.choose_move(temperature, &mut self.rng);
        let legal_moves: Vec<_> = policy.iter().map(|entry| entry.mv).collect();
        self.positions.push(Position {
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
        self.game.play_legal(played);
        self.search = Search::new();
        self.game.outcome()
    }
}

fn play_batch(
    model: &Hybrid,
    settings: &Settings,
    config: &Config,
    ids: std::ops::Range<usize>,
) -> Result<Vec<(usize, PlayedGame)>> {
    let visits = NonZeroU32::from(config.visits_per_move);
    let mut active: Vec<_> = ids.map(|id| ActiveGame::new(config.seed, id)).collect();
    let mut completed = Vec::with_capacity(active.len());
    while !active.is_empty() {
        let mut pending = Vec::new();
        let mut finished = Vec::new();
        for (i, slot) in active.iter_mut().enumerate() {
            if let Some(leaf) = slot
                .search
                .next_leaf(&slot.game, visits, settings.exploration)
            {
                pending.push((i, leaf));
            } else if let Some(outcome) = slot.play(config.temperature) {
                let white_value = match outcome {
                    Outcome::Win(Color::White) => 1,
                    Outcome::Win(Color::Black) => -1,
                    Outcome::Draw => 0,
                };
                finished.push((i, white_value));
            }
        }
        let evaluations = model.evaluate(&pending)?;
        for ((i, leaf), result) in pending.into_iter().zip(evaluations) {
            active[i]
                .search
                .evaluate(leaf, result.value, &result.priors);
        }
        for (i, white_value) in finished.into_iter().rev() {
            let slot = active.swap_remove(i);
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

pub fn generate(config: &Config, model: &Hybrid, settings: &Settings) -> Result<PathBuf> {
    if !config.temperature.is_finite() || config.temperature < 0.0 {
        return Err("self-play temperature must be finite and nonnegative".into());
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
        let mut games = Vec::with_capacity(end - start);
        for batch_start in (start..end).step_by(config.batch.get()) {
            let batch_end = batch_start.saturating_add(config.batch.get()).min(end);
            games.extend(play_batch(model, settings, config, batch_start..batch_end)?);
        }
        let elapsed = started.elapsed();
        let shard_positions: usize = games.iter().map(|(_, game)| game.positions.len()).sum();
        let generation_ns_per_position =
            u64::try_from(elapsed.as_nanos() / shard_positions as u128)?;
        write_shard(
            &run,
            shards,
            games
                .iter()
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
        "evaluator": "piece-count-wdl-v1",
        "priors": "smolgen-policy-v1",
        "checkpoint": model.model_dir,
        "device": config.device,
        "game_batch": config.batch.get(),
        "sampling": "chacha8-stream-per-game; temperature applied to root visits",
        "temperature": config.temperature,
        "exploration": settings.exploration,
    });
    let partial = run.join("manifest.json.partial");
    fs::write(&partial, format!("{manifest:#}\n"))?;
    fs::rename(partial, run.join("manifest.json"))?;
    Ok(run)
}
