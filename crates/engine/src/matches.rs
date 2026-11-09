use std::{
    collections::BTreeMap,
    io,
    num::{NonZeroU16, NonZeroU32},
    path::PathBuf,
};

use rand::SeedableRng;
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

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Options {
    pub baseline_dir: PathBuf,
    pub device: String,
    batch: u16,
    leaves_per_game: NonZeroU16,
    #[serde(rename = "visits_per_move")]
    visits: NonZeroU32,
    max_pairs: u64,
    seed: u64,
    sprt: sprt::Config,
}

pub fn run(options: Options, players: [&Hybrid; 2], settings: &Settings) -> io::Result<()> {
    if options.batch == 0 || options.max_pairs == 0 || options.max_pairs > u64::MAX / 2 {
        return Err(io::Error::other(
            "batch must be positive and max pairs must be positive and fit twice in u64",
        ));
    }
    if !options.sprt.validate() {
        return Err(io::Error::other("invalid SPRT Elo bounds or error rates"));
    }
    let (lower, upper) = options.sprt.boundaries();
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
        "checkpoint match; SPRT [{}, {}] logistic Elo, alpha={} beta={}; batch={} leaves_per_game={} visits={} max_pairs={} seed={}",
        options.sprt.lower_elo,
        options.sprt.upper_elo,
        options.sprt.alpha,
        options.sprt.beta,
        options.batch,
        options.leaves_per_game,
        options.visits,
        options.max_pairs,
        options.seed
    );

    while !active.is_empty() {
        let mut pending: [Vec<(usize, Leaf)>; 2] = [Vec::new(), Vec::new()];
        let mut finished = Vec::new();
        for (i, slot) in active.iter_mut().enumerate() {
            let batch = &mut pending[slot.player()];
            let before = batch.len();
            for _ in 0..options.leaves_per_game.get() {
                let Some(leaf) =
                    slot.search
                        .next_leaf(&slot.game, options.visits, settings.exploration)
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
            llr = options.sprt.llr(counts);
            let decided = llr <= lower || llr >= upper;
            if next_pair.is_multiple_of(10) || decided {
                println!(
                    "pairs={next_pair} candidate W/D/L={wdl:?} pentanomial={counts:?} LLR={llr:.4}"
                );
            }
            if decided {
                let (hypothesis, elo) = if llr >= upper {
                    ("H1", options.sprt.upper_elo)
                } else {
                    ("H0", options.sprt.lower_elo)
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
        "inconclusive at cap: pairs={next_pair} candidate W/D/L={wdl:?} pentanomial={counts:?} LLR={llr:.4}; largest inference batches (baseline/candidate)={max_batches:?}"
    );
    Ok(())
}
