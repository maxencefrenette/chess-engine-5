use std::{
    env, fs,
    num::{NonZeroU32, NonZeroUsize},
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use chess_data::{FORMAT_VERSION, Position, Result, RunWriter, VISITS_PER_MOVE};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde_json::json;

use crate::{
    chess::{Color, Game, Outcome},
    hybrid::Hybrid,
    mcts::Search,
    settings::Settings,
};

const VISITS: NonZeroU32 = NonZeroU32::new(VISITS_PER_MOVE as u32).unwrap();

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
    seed: u64,
    ids: std::ops::Range<usize>,
) -> Result<Vec<(usize, PlayedGame)>> {
    let mut active: Vec<_> = ids.map(|id| ActiveGame::new(seed, id)).collect();
    let mut completed = Vec::with_capacity(active.len());
    while !active.is_empty() {
        let mut pending = Vec::new();
        let mut finished = Vec::new();
        for (i, slot) in active.iter_mut().enumerate() {
            if let Some(leaf) = slot
                .search
                .next_leaf(&slot.game, VISITS, settings.exploration)
            {
                pending.push((i, leaf));
            } else if let Some(outcome) = slot.play(settings.self_play_temperature) {
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
                .evaluate(leaf.moves, result.value, &result.priors);
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

pub fn generate(
    root: &Path,
    games: NonZeroUsize,
    seed: u64,
    batch: NonZeroUsize,
    model: &Hybrid,
    settings: &Settings,
) -> Result<PathBuf> {
    let version = root.join(format!("v{FORMAT_VERSION}"));
    fs::create_dir_all(&version)?;
    let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let run = version.join(format!("{now}-{}", std::process::id()));
    fs::create_dir(&run)?;
    eprintln!("Writing self-play run to {}", run.display());

    let mut writer = RunWriter::new(&run);
    let mut positions = 0;
    let mut white_wins = 0;
    let mut black_wins = 0;
    let mut draws = 0;
    for start in (0..games.get()).step_by(batch.get()) {
        let end = start.saturating_add(batch.get()).min(games.get());
        for (id, game) in play_batch(model, settings, seed, start..end)? {
            writer.write_game(id as u64, &game.positions, game.white_value)?;
            positions += game.positions.len();
            match game.white_value {
                1 => white_wins += 1,
                -1 => black_wins += 1,
                _ => draws += 1,
            }
        }
        eprintln!("{end}/{} games, {positions} positions", games.get());
    }
    let shards = writer.finish()?;

    let manifest = json!({
        "format_version": FORMAT_VERSION,
        "seed": seed,
        "engine_version": env!("CARGO_PKG_VERSION"),
        "engine_revision": env::var("CE5_ENGINE_REVISION").ok(),
        "games": games.get(),
        "positions": positions,
        "shards": shards,
        "white_wins": white_wins,
        "black_wins": black_wins,
        "draws": draws,
        "start_position": "standard",
        "visits_per_move": VISITS.get(),
        "evaluator": "piece-count-wdl-v1",
        "priors": "smolgen-policy-v1",
        "checkpoint": model.model_dir,
        "game_batch": batch.get(),
        "sampling": "chacha8-stream-per-game; temperature applied to root visits",
        "temperature": settings.self_play_temperature,
        "exploration": settings.exploration,
    });
    let partial = run.join("manifest.json.partial");
    fs::write(&partial, format!("{manifest:#}\n"))?;
    fs::rename(partial, run.join("manifest.json"))?;
    Ok(run)
}
