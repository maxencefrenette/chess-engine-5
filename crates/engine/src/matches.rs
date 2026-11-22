use std::{
    io,
    num::{NonZeroU16, NonZeroU32, NonZeroUsize},
    path::PathBuf,
    sync::Mutex,
    time::{Duration, Instant},
};

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::{
    chess::{Color, Game, Outcome},
    mcts::Search,
    model_evaluator::ModelEvaluator,
    settings::SearchSettings,
    sprt,
};

pub mod gauntlet;
mod search;
use search::{Group, Work};

struct ActiveGame {
    id: u64,
    game: Game,
    // Each player can reuse only predictions from its own searches.
    searches: [Search; 2],
}

impl ActiveGame {
    fn new(id: u64, openings: &[Game]) -> Self {
        Self {
            id,
            game: openings[(id / 2) as usize].clone(),
            searches: std::array::from_fn(|_| Search::new()),
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

    fn play(&mut self, settings: &[SearchSettings; 2]) -> Option<Outcome> {
        let player = self.player();
        let mv = self.searches[player].best_move();
        self.game.play_legal(mv);
        let outcome = self.game.outcome();
        if outcome.is_none() {
            for (side, tree) in self.searches.iter_mut().enumerate() {
                if settings[side].reuse_subtree {
                    tree.advance_root(mv);
                } else if side == player {
                    *tree = Search::new();
                }
            }
        }
        outcome
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
    #[range(min = 0.0)]
    pub loss_stddev: f64,
    batch: u16,
    search_threads: NonZeroUsize,
    leaves_per_game: NonZeroU16,
    baseline_visits_per_move: NonZeroU32,
    max_pairs: u64,
    min_passed_gainer_pairs: u64,
    seed: u64,
    sprt: SprtPresets,
}

fn nelo_summary(counts: [u64; 5], sprt: Option<&sprt::Config>) -> String {
    let (estimate, method) = match sprt {
        Some(config) => (config.estimate(counts), "SPRT-adjusted approximate 95% CI"),
        None => (
            sprt::fixed_estimate(counts),
            "fixed-sample approximate 95% CI",
        ),
    };
    match estimate {
        Some(sprt::Estimate {
            nelo,
            interval: [lower, upper],
        }) => {
            format!("{nelo:+.1} ({lower:+.1}, {upper:+.1}) [{method}]")
        }
        None => "unavailable (insufficient pair-score variation)".into(),
    }
}

fn time_matched_visits(
    settings: &SearchSettings,
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
        let Some(leaf) = Search::new().next_leaf(&game, NonZeroU32::MIN, settings) else {
            game = Game::default();
            continue;
        };
        game.play_legal(leaf.moves[rng.random_range(0..leaf.moves.len())]);
        batch.push(leaf);
    }

    let measure = |player: &ModelEvaluator| -> io::Result<Duration> {
        let inputs = batch.iter().map(|leaf| leaf.input()).collect();
        let start = Instant::now();
        player.evaluate(inputs)?;
        Ok(start.elapsed())
    };
    for _ in 0..4 {
        for player in players {
            measure(player)?;
        }
    }
    let mut timings = [[Duration::ZERO; 2]; SAMPLES];
    for (sample, timing) in timings.iter_mut().enumerate() {
        for offset in 0..2 {
            let side = (sample + offset) % 2;
            timing[side] = measure(players[side])?;
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
    settings: [SearchSettings; 2],
) -> io::Result<()> {
    let openings = load_openings(&options)?;
    play(&options, &openings, mode, matched_speed, players, settings)?;
    Ok(())
}

struct MatchResult {
    wdl: [u64; 3],
    counts: [u64; 5],
    candidate_visits: NonZeroU32,
}

fn load_openings(options: &Options) -> io::Result<Vec<Game>> {
    if options.batch == 0 || options.max_pairs == 0 || options.max_pairs > u64::MAX / 2 {
        return Err(io::Error::other(
            "batch must be positive and max pairs must be positive and fit twice in u64",
        ));
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
    Ok(openings)
}

fn play(
    options: &Options,
    openings: &[Game],
    mode: Mode,
    matched_speed: bool,
    players: [&ModelEvaluator; 2],
    settings: [SearchSettings; 2],
) -> io::Result<MatchResult> {
    let (mut sprt, pass_floor) = match mode {
        Mode::Gainer => (
            Some(&options.sprt.gainer),
            options.min_passed_gainer_pairs.min(options.max_pairs),
        ),
        Mode::NonRegression => (Some(&options.sprt.non_regression), 0),
        Mode::Fixed => (None, 0),
    };
    if sprt.as_ref().is_some_and(|sprt| !sprt.validate()) {
        return Err(io::Error::other("invalid SPRT nElo bounds or error rates"));
    }
    let limit = 2 * options.max_pairs;
    let slots = u64::from(options.batch).min(limit) as usize;
    // Benchmark a full worker group. The queue can combine several ready groups.
    let games_per_group = slots.div_ceil(options.search_threads.get()).div_ceil(2);
    let benchmark_batch = games_per_group * usize::from(options.leaves_per_game.get());
    let visits = if std::ptr::eq(players[0], players[1]) {
        [options.baseline_visits_per_move; 2]
    } else {
        time_matched_visits(
            &settings[0],
            players,
            benchmark_batch,
            options.baseline_visits_per_move,
            matched_speed,
        )?
    };
    for (name, search) in ["baseline", "candidate"].into_iter().zip(&settings) {
        println!(
            "{name} search: exploration={} reuse_subtree={} first_play={:?} pending_value={:?} policy_temperature={}",
            search.exploration,
            search.reuse_subtree,
            search.first_play,
            search.pending_value,
            search.policy_temperature
        );
    }
    let mut work = Mutex::new(Work {
        ids: 0..limit,
        results: Default::default(),
    });
    let mut next_pair = 0;
    let mut counts = [0; 5];
    let mut wdl = [0; 3];
    let mut llr = 0.0;
    let mut early_pass = None;
    let mut max_batches = [0; 2];
    if let Some(sprt) = &sprt {
        println!(
            "SPRT [{}, {}] nElo, alpha={} beta={}",
            sprt.lower_nelo, sprt.upper_nelo, sprt.alpha, sprt.beta
        );
    }
    println!(
        "batch match; batch={} leaves_per_game={} visits baseline={} candidate={} max_pairs={} seed={}",
        options.batch,
        options.leaves_per_game,
        visits[0],
        visits[1],
        options.max_pairs,
        options.seed
    );

    let mut accepted = None;
    let mut score = |work: &mut Work| -> io::Result<bool> {
        while accepted.is_none() && next_pair < work.ids.end / 2 {
            let Some([first, second]) = work.take_pair(next_pair) else {
                return Ok(false);
            };
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
                    Some(("H1", sprt.upper_nelo))
                } else if llr <= lower {
                    Some(("H0", sprt.lower_nelo))
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
                    "pairs={next_pair} candidate W/D/L={wdl:?} nElo={} pentanomial={counts:?}{evidence}",
                    nelo_summary(counts, sprt)
                );
            }
            if let Some((hypothesis, nelo)) = decision {
                if hypothesis == "H1" && next_pair < pass_floor {
                    println!("gainer passed at pair {next_pair}; continuing to {pass_floor} pairs");
                    early_pass = Some((next_pair, nelo));
                    sprt = None;
                    work.ids.end = 2 * pass_floor;
                } else {
                    accepted = Some((hypothesis, nelo));
                }
            }
        }
        Ok(true)
    };
    crate::inference_queue::run(
        slots,
        options.search_threads,
        |n| Group::new(n, &work, openings),
        |group, predictions| {
            group.advance(
                predictions,
                &work,
                openings,
                visits,
                &settings,
                options.leaves_per_game,
            )
        },
        |side, inputs| {
            if score(&mut work.lock().unwrap())? {
                return Ok(None);
            }
            max_batches[side] = max_batches[side].max(inputs.len());
            players[side].evaluate(inputs).map(Some)
        },
    )?;
    if !score(work.get_mut().unwrap())? {
        return Err(io::Error::other("match ended with incomplete game pairs"));
    }
    let result = MatchResult {
        wdl,
        counts,
        candidate_visits: visits[1],
    };
    if let Some((hypothesis, nelo)) = accepted {
        println!(
            "accepted {hypothesis} ({nelo:+} nElo); largest inference batches (baseline/candidate)={max_batches:?}"
        );
        return Ok(result);
    }
    let status = if let Some((pair, nelo)) = early_pass {
        format!("accepted H1 ({nelo:+} nElo) at pair {pair}")
    } else if sprt.is_some() {
        "inconclusive at cap".into()
    } else {
        "fixed match complete".into()
    };
    let evidence = if sprt.is_some() {
        format!(" LLR={llr:.4}")
    } else {
        String::new()
    };
    println!(
        "{status}: pairs={next_pair} candidate W/D/L={wdl:?} nElo={} pentanomial={counts:?}{evidence}; largest inference batches (baseline/candidate)={max_batches:?}",
        nelo_summary(counts, sprt)
    );
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn reporting_uses_nelo_for_estimates_and_sprt_bounds() -> Result<(), String> {
        let counts = [135, 440, 682, 529, 214];
        let config = sprt::Config {
            lower_nelo: 0.0,
            upper_nelo: 20.0,
            alpha: 0.05,
            beta: 0.05,
        };
        assert_eq!(
            nelo_summary(counts, Some(&config)),
            "+24.8 (+1.7, +47.2) [SPRT-adjusted approximate 95% CI]"
        );
        assert_eq!(
            nelo_summary(counts, None),
            "+28.1 (+17.3, +38.9) [fixed-sample approximate 95% CI]"
        );
        for counts in [
            [0; 5],
            [0, 0, 1, 0, 0],
            [0, 0, 100, 0, 0],
            [100, 0, 0, 0, 0],
        ] {
            assert_eq!(
                nelo_summary(counts, None),
                "unavailable (insufficient pair-score variation)"
            );
            assert_eq!(
                nelo_summary(counts, Some(&config)),
                "unavailable (insufficient pair-score variation)"
            );
        }
        Ok(())
    }

    #[test]
    fn configured_baseline_checkpoint_loads_on_cpu() -> Result<(), Box<dyn std::error::Error>> {
        use chess_config::TomlConfig;
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
