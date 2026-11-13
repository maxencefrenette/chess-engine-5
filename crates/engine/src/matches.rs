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
    mcts::{Leaf, Search},
    model_evaluator::ModelEvaluator,
    settings::Settings,
    sprt,
};

struct ActiveGame {
    id: u64,
    game: Game,
    search: Search,
}

impl ActiveGame {
    fn new(id: u64, openings: &[Game]) -> Self {
        Self {
            id,
            game: openings[(id / 2) as usize].clone(),
            search: Search::new(),
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

    fn play(&mut self) -> Option<Outcome> {
        self.game.play_legal(self.search.best_move());
        self.search = Search::new();
        self.game.outcome()
    }
}

pub enum Mode {
    Gainer,
    NonRegression,
    Fixed,
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
    opening_book: String,
    pub device: chess_config::DeviceConfig,
    pub loss_stddev: f64,
    batch: u16,
    leaves_per_game: NonZeroU16,
    baseline_visits_per_move: NonZeroU32,
    max_pairs: u64,
    seed: u64,
    sprt: SprtPresets,
}

fn elo_summary(counts: [u64; 5], sprt: Option<&sprt::Config>) -> String {
    let (estimate, method) = match sprt {
        Some(config) => (config.estimate(counts), "SPRT-adjusted approximate 95% CI"),
        None => (sprt::fixed_estimate(counts), "approximate 95% CI"),
    };
    match estimate {
        Some(sprt::Estimate {
            elo,
            interval: [lower, upper],
        }) => {
            format!("{elo:+.1} ({lower:+.1}, {upper:+.1}) [{method}]")
        }
        None => "unavailable (insufficient pair-score variation)".into(),
    }
}

fn time_matched_visits(
    players: [&ModelEvaluator; 2],
    batch_size: usize,
    baseline_visits: NonZeroU32,
    matched_speed: bool,
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
    let candidate_visits = if matched_speed {
        validate_matched_speed(timings.map(|pair| pair.map(|duration| duration.as_secs_f64())))?;
        baseline_visits
    } else {
        NonZeroU32::new(scaled).unwrap_or(NonZeroU32::MIN)
    };
    println!(
        "inference benchmark: batch={batch_size} median baseline={:.3} ms candidate={:.3} ms; visits baseline={baseline_visits} candidate={candidate_visits}",
        baseline * 1000.0,
        candidate * 1000.0
    );
    Ok([baseline_visits, candidate_visits])
}

fn validate_matched_speed(timings: [[f64; 2]; 31]) -> io::Result<()> {
    if timings
        .iter()
        .flatten()
        .any(|t| !t.is_finite() || *t <= 0.0)
    {
        return Err(io::Error::other(
            "inference benchmark needs finite positive timings",
        ));
    }
    // Paired log ratios control for shared drift and treat speedups and slowdowns symmetrically.
    let ratios = timings.map(|[baseline, candidate]| (candidate / baseline).ln());
    let mean = ratios.iter().sum::<f64>() / ratios.len() as f64;
    let variance = ratios
        .iter()
        .map(|ratio| (ratio - mean).powi(2))
        .sum::<f64>()
        / (ratios.len() - 1) as f64;
    // Two-sided 95% Student-t interval, 30 degrees of freedom.
    let margin = 2.042272456 * (variance / ratios.len() as f64).sqrt();
    let [lower, upper] = [mean - margin, mean + margin].map(f64::exp);
    let summary = format!(
        "candidate/baseline inference time ratio={:.4}, 95% CI [{lower:.4}, {upper:.4}]",
        mean.exp()
    );
    if lower > 1.0 || upper < 1.0 {
        return Err(io::Error::other(format!(
            "--matched-speed failed: {summary}; speed difference exceeds benchmark variability"
        )));
    }
    println!("matched-speed check passed: {summary}");
    Ok(())
}

pub fn run(
    options: Options,
    mode: Mode,
    matched_speed: bool,
    players: [&ModelEvaluator; 2],
    settings: &Settings,
) -> io::Result<()> {
    if options.batch == 0 || options.max_pairs == 0 || options.max_pairs > u64::MAX / 2 {
        return Err(io::Error::other(
            "batch must be positive and max pairs must be positive and fit twice in u64",
        ));
    }
    let sprt = match mode {
        Mode::Gainer => Some(options.sprt.gainer),
        Mode::NonRegression => Some(options.sprt.non_regression),
        Mode::Fixed => None,
    };
    if sprt.as_ref().is_some_and(|sprt| !sprt.validate()) {
        return Err(io::Error::other("invalid SPRT Elo bounds or error rates"));
    }
    let openings = chess_opening_book::load(
        &options.opening_book,
        options.max_pairs as usize,
        options.seed,
    )?
    .into_iter()
    .map(|fen| {
        let game = Game::from_fen(&fen)
            .map_err(|error| io::Error::other(format!("opening {fen}: {error}")))?;
        if game.outcome().is_some() {
            return Err(io::Error::other(format!("terminal opening: {fen}")));
        }
        Ok(game)
    })
    .collect::<io::Result<Vec<_>>>()?;
    println!(
        "opening_book={} pairs={}",
        options.opening_book,
        openings.len()
    );
    let benchmark_batch = usize::from(options.batch) * usize::from(options.leaves_per_game.get());
    let visits = time_matched_visits(
        players,
        benchmark_batch,
        options.baseline_visits_per_move,
        matched_speed,
    )?;
    let limit = 2 * options.max_pairs;
    let mut next_id = u64::from(options.batch).min(limit);
    let mut active: Vec<_> = (0..next_id)
        .map(|id| ActiveGame::new(id, &openings))
        .collect();
    let mut results = BTreeMap::new();
    let mut next_pair = 0;
    let mut counts = [0; 5];
    let mut wdl = [0; 3];
    let mut llr = 0.0;
    let mut max_batches = [0; 2];
    if let Some(sprt) = &sprt {
        println!(
            "SPRT [{}, {}] normalized Elo, alpha={} beta={}",
            sprt.lower_elo, sprt.upper_elo, sprt.alpha, sprt.beta
        );
    }
    println!(
        "checkpoint match; batch={} leaves_per_game={} visits baseline={} candidate={} max_pairs={} seed={}",
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
                && let Some(outcome) = slot.play()
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
            if let Some(sprt) = &sprt {
                llr = sprt.llr(counts)?;
            }
            let decision = sprt.as_ref().and_then(|sprt| {
                let (lower, upper) = sprt.boundaries();
                if llr >= upper {
                    Some(("H1", sprt.upper_elo))
                } else if llr <= lower {
                    Some(("H0", sprt.lower_elo))
                } else {
                    None
                }
            });
            if next_pair.is_multiple_of(10) || decision.is_some() {
                let evidence = if sprt.is_some() {
                    format!(" LLR={llr:.4}")
                } else {
                    String::new()
                };
                println!(
                    "pairs={next_pair} candidate W/D/L={wdl:?} Elo={} pentanomial={counts:?}{evidence}",
                    elo_summary(counts, sprt.as_ref())
                );
            }
            if let Some((hypothesis, elo)) = decision {
                println!(
                    "accepted {hypothesis} ({elo:+} normalized Elo); largest inference batches (baseline/candidate)={max_batches:?}"
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
                active[i] = ActiveGame::new(next_id, &openings);
                next_id += 1;
            } else {
                active.swap_remove(i);
            }
        }
    }
    let status = if sprt.is_some() {
        "inconclusive at cap"
    } else {
        "fixed match complete"
    };
    let evidence = if sprt.is_some() {
        format!(" LLR={llr:.4}")
    } else {
        String::new()
    };
    println!(
        "{status}: pairs={next_pair} candidate W/D/L={wdl:?} Elo={} pentanomial={counts:?}{evidence}; largest inference batches (baseline/candidate)={max_batches:?}",
        elo_summary(counts, sprt.as_ref())
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use chess_config::TomlConfig;

    #[test]
    fn matched_speed_accepts_noise_and_rejects_both_directions() {
        let timings = std::array::from_fn(|i| {
            let baseline = 0.001 * (1.0 + i as f64 / 100.0);
            [baseline, baseline * ((i as f64 - 15.0) * 0.001).exp()]
        });
        assert!(validate_matched_speed(timings).is_ok());
        assert!(validate_matched_speed([[0.001, 0.001]; 31]).is_ok());
        for factor in [0.9, 1.1] {
            assert!(validate_matched_speed(timings.map(|[a, b]| [a, b * factor])).is_err());
        }
        assert!(validate_matched_speed([[0.0, 0.001]; 31]).is_err());
    }

    #[test]
    fn paired_games_start_from_the_same_opening_with_colors_swapped() {
        let opening = Game::from_fen("r3k2r/8/8/8/3pP3/8/8/R3K2R b KQkq e3 0 15").unwrap();
        let openings = [opening];
        let mut first = ActiveGame::new(0, &openings);
        let second = ActiveGame::new(1, &openings);
        assert_eq!(first.game.turn(), Color::Black);
        assert_eq!(second.game.turn(), Color::Black);
        assert_eq!(first.player(), 0);
        assert_eq!(second.player(), 1);
        assert_eq!(
            first.game.snapshot(&first.game.legal_moves()),
            second.game.snapshot(&second.game.legal_moves())
        );
        let mv = first.game.legal_moves()[0];
        first.game.play_legal(mv);
        assert_eq!(second.game.turn(), Color::Black);
        assert_eq!(openings[0].turn(), Color::Black);
    }

    #[test]
    fn reporting_uses_ordinary_elo_with_normalized_sprt_bounds() {
        let counts = [135, 440, 682, 529, 214];
        let config = sprt::Config {
            lower_elo: 0.0,
            upper_elo: 20.0,
            alpha: 0.05,
            beta: 0.05,
        };
        assert_eq!(
            elo_summary(counts, Some(&config)),
            "+18.9 (+1.3, +36.2) [SPRT-adjusted approximate 95% CI]"
        );
        assert_eq!(
            elo_summary(counts, None),
            "+21.5 (+13.2, +29.8) [approximate 95% CI]"
        );
        for counts in [
            [0; 5],
            [0, 0, 1, 0, 0],
            [0, 0, 100, 0, 0],
            [100, 0, 0, 0, 0],
        ] {
            assert_eq!(
                elo_summary(counts, None),
                "unavailable (insufficient pair-score variation)"
            );
            assert_eq!(
                elo_summary(counts, Some(&config)),
                "unavailable (insufficient pair-score variation)"
            );
        }
    }

    #[test]
    fn configured_baseline_checkpoint_loads_on_cpu() -> Result<(), Box<dyn std::error::Error>> {
        let options = Options::from_repo_file("eval-model.toml")?;
        ModelEvaluator::load(&options.baseline_dir, chess_config::DeviceConfig::Cpu).map_err(
            |error| {
                format!(
                    "loading baseline {}: {error}",
                    options.baseline_dir.display()
                )
            },
        )?;
        Ok(())
    }
}
