use std::{
    collections::BTreeMap,
    io,
    num::{NonZeroU16, NonZeroU32},
    path::PathBuf,
    time::{Duration, Instant},
};

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::{
    chess::{Color, Game, Outcome},
    hybrid::Hybrid,
    mcts::{Leaf, Search},
    settings::Settings,
    sprt,
};

struct ActiveGame {
    id: u64,
    game: Game,
    search: Search,
    rng: ChaCha8Rng,
    ply: u32,
}

impl ActiveGame {
    fn new(id: u64, seed: u64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        rng.set_stream(id / 2);
        Self {
            id,
            game: Game::default(),
            search: Search::new(),
            rng,
            ply: 0,
        }
    }

    fn candidate_color(&self) -> Color {
        if self.id.is_multiple_of(2) {
            Color::White
        } else {
            Color::Black
        }
    }

    fn player(&self) -> usize {
        usize::from(self.game.turn() == self.candidate_color())
    }

    fn play(&mut self, settings: &Settings) -> Option<Outcome> {
        let temperature = if self.ply < settings.eval_sampling_plies {
            settings.eval_temperature
        } else {
            0.0
        };
        self.game
            .play_legal(self.search.choose_move(temperature, &mut self.rng));
        self.ply += 1;
        self.search = Search::new();
        self.game.outcome()
    }
}

pub enum Mode {
    Gainer,
    NonRegression,
}

#[chess_config::toml_config]
struct SprtPresets {
    gainer: sprt::Config,
    #[serde(rename = "non-regression")]
    non_regression: sprt::Config,
}

#[chess_config::toml_config]
pub struct Options {
    pub baseline_dir: PathBuf,
    pub device: chess_config::DeviceConfig,
    batch: u16,
    leaves_per_game: NonZeroU16,
    baseline_visits_per_move: NonZeroU32,
    max_pairs: u64,
    seed: u64,
    sprt: SprtPresets,
}

fn observed_elo([wins, draws, losses]: [u64; 3]) -> f64 {
    let half_draws = draws as f64 / 2.0;
    400.0 * ((wins as f64 + half_draws) / (losses as f64 + half_draws)).log10()
}

fn time_matched_visits(
    players: [&Hybrid; 2],
    batch_size: usize,
    baseline_visits: NonZeroU32,
) -> io::Result<[NonZeroU32; 2]> {
    const SAMPLES: usize = 31;
    let mut rng = ChaCha8Rng::seed_from_u64(0);
    let mut game = Game::default();
    let mut batch = Vec::with_capacity(batch_size);
    while batch.len() < batch_size {
        // Use the same nonterminal positions and legal moves for both checkpoints.
        let Some(leaf) = Search::new().next_leaf(&game, NonZeroU32::MIN, 0.0) else {
            game = Game::default();
            continue;
        };
        game.play_legal(leaf.moves[rng.random_range(0..leaf.moves.len())]);
        batch.push((batch.len(), leaf));
    }

    for _ in 0..4 {
        for player in players {
            player.evaluate(&batch)?;
        }
    }
    let mut timings = [[Duration::ZERO; 2]; SAMPLES];
    for (sample, timing) in timings.iter_mut().enumerate() {
        for offset in 0..2 {
            let side = (sample + offset) % 2;
            let start = Instant::now();
            players[side].evaluate(&batch)?;
            timing[side] = start.elapsed();
        }
    }
    let [baseline, candidate] = [0, 1].map(|side| {
        let mut samples = timings.map(|sample| sample[side]);
        samples.sort_unstable();
        samples[SAMPLES / 2].as_secs_f64()
    });
    if baseline == 0.0 || candidate == 0.0 {
        return Err(io::Error::other(
            "inference benchmark measured zero elapsed time",
        ));
    }
    let scaled = (f64::from(baseline_visits.get()) * baseline / candidate).round() as u32;
    let candidate_visits = NonZeroU32::new(scaled).unwrap_or(NonZeroU32::MIN);
    println!(
        "inference benchmark: batch={batch_size} median baseline={:.3} ms candidate={:.3} ms; visits baseline={baseline_visits} candidate={candidate_visits}",
        baseline * 1000.0,
        candidate * 1000.0
    );
    Ok([baseline_visits, candidate_visits])
}

pub fn run(
    options: Options,
    mode: Mode,
    players: [&Hybrid; 2],
    settings: &Settings,
) -> io::Result<()> {
    if options.batch == 0 || options.max_pairs == 0 || options.max_pairs > u64::MAX / 2 {
        return Err(io::Error::other(
            "batch must be positive and max pairs must be positive and fit twice in u64",
        ));
    }
    let sprt = match mode {
        Mode::Gainer => options.sprt.gainer,
        Mode::NonRegression => options.sprt.non_regression,
    };
    if !sprt.validate() {
        return Err(io::Error::other("invalid SPRT Elo bounds or error rates"));
    }
    let benchmark_batch = usize::from(options.batch) * usize::from(options.leaves_per_game.get());
    let visits = time_matched_visits(players, benchmark_batch, options.baseline_visits_per_move)?;
    let (lower, upper) = sprt.boundaries();
    let limit = 2 * options.max_pairs;
    let mut next_id = u64::from(options.batch).min(limit);
    let mut active: Vec<_> = (0..next_id)
        .map(|id| ActiveGame::new(id, options.seed))
        .collect();
    let mut results = BTreeMap::new();
    let mut next_pair = 0;
    let mut counts = [0; 5];
    let mut wdl = [0; 3];
    let mut llr = 0.0;
    let mut max_batches = [0; 2];
    println!(
        "checkpoint match; SPRT [{}, {}] logistic Elo, alpha={} beta={}; batch={} leaves_per_game={} visits baseline={} candidate={} max_pairs={} seed={}",
        sprt.lower_elo,
        sprt.upper_elo,
        sprt.alpha,
        sprt.beta,
        options.batch,
        options.leaves_per_game,
        visits[0],
        visits[1],
        options.max_pairs,
        options.seed
    );

    while !active.is_empty() {
        let mut pending: [Vec<(usize, Leaf)>; 2] = [Vec::new(), Vec::new()];
        let mut finished = Vec::new();
        for (i, slot) in active.iter_mut().enumerate() {
            let side = slot.player();
            let batch = &mut pending[side];
            let before = batch.len();
            for _ in 0..options.leaves_per_game.get() {
                let Some(leaf) =
                    slot.search
                        .next_leaf(&slot.game, visits[side], settings.exploration)
                else {
                    break;
                };
                batch.push((i, leaf));
            }
            if batch.len() == before
                && let Some(outcome) = slot.play(settings)
            {
                let points = match outcome {
                    Outcome::Win(color) if color == slot.candidate_color() => 2,
                    Outcome::Win(_) => 0,
                    Outcome::Draw => 1,
                };
                results.insert(slot.id, points);
                finished.push(i);
            }
        }

        while let (Some(&first), Some(&second)) = (
            results.get(&(2 * next_pair)),
            results.get(&(2 * next_pair + 1)),
        ) {
            results.remove(&(2 * next_pair));
            results.remove(&(2 * next_pair + 1));
            counts[first + second] += 1;
            wdl[2 - first] += 1;
            wdl[2 - second] += 1;
            next_pair += 1;
            llr = sprt.llr(counts);
            let decided = llr <= lower || llr >= upper;
            if next_pair.is_multiple_of(10) || decided {
                println!(
                    "pairs={next_pair} candidate W/D/L={wdl:?} observed Elo={:+.1} pentanomial={counts:?} LLR={llr:.4}",
                    observed_elo(wdl)
                );
            }
            if decided {
                let (hypothesis, elo) = if llr >= upper {
                    ("H1", sprt.upper_elo)
                } else {
                    ("H0", sprt.lower_elo)
                };
                println!(
                    "accepted {hypothesis} ({elo:+} Elo); largest inference batches (baseline/candidate)={max_batches:?}"
                );
                return Ok(());
            }
        }

        for (side, batch) in pending.into_iter().enumerate() {
            max_batches[side] = max_batches[side].max(batch.len());
            let evaluations = players[side].evaluate(&batch)?;
            for ((slot, leaf), result) in batch.into_iter().zip(evaluations) {
                active[slot]
                    .search
                    .evaluate(leaf, result.value, &result.priors);
            }
        }
        for i in finished.into_iter().rev() {
            if next_id < limit {
                active[i] = ActiveGame::new(next_id, options.seed);
                next_id += 1;
            } else {
                active.swap_remove(i);
            }
        }
    }
    println!(
        "inconclusive at cap: pairs={next_pair} candidate W/D/L={wdl:?} observed Elo={:+.1} pentanomial={counts:?} LLR={llr:.4}; largest inference batches (baseline/candidate)={max_batches:?}",
        observed_elo(wdl)
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use chess_config::TomlConfig;

    #[test]
    fn configured_baseline_checkpoint_loads_on_cpu() -> Result<(), Box<dyn std::error::Error>> {
        let options = Options::from_file(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../config/eval-model.toml"
        ))?;
        Hybrid::load(&options.baseline_dir, chess_config::DeviceConfig::Cpu).map_err(|error| {
            format!(
                "loading baseline {}: {error}",
                options.baseline_dir.display()
            )
        })?;
        Ok(())
    }
}
