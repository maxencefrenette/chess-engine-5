use std::{
    env,
    num::{NonZeroU32, NonZeroUsize},
    path::Path,
    process::ExitCode,
};

use chess_engine::{
    chess::Game,
    matches::{self, Options},
    mcts::{material_value, search},
    self_play,
};

fn run() -> Result<(), String> {
    let mut args = env::args().skip(1);
    let first = args
        .next()
        .ok_or("usage: chess-engine SIMULATIONS [MOVE ...] | self-play OUTPUT_DIR GAMES SEED | match [BATCH] [VISITS] [MAX_PAIRS] [SEED] | match-hybrid MODEL_DIR VALUE_JSON [BATCH] [VISITS] [MAX_PAIRS] [SEED] [DEVICE]")?;
    if first == "match" || first == "match-hybrid" {
        let hybrid = if first == "match-hybrid" {
            Some((
                args.next().ok_or("missing policy model directory")?,
                args.next().ok_or("missing value JSON")?,
            ))
        } else {
            None
        };
        let batch = args
            .next()
            .as_deref()
            .unwrap_or("8")
            .parse()
            .map_err(|_| "invalid batch")?;
        let visits = args
            .next()
            .as_deref()
            .unwrap_or("800")
            .parse()
            .map_err(|_| "invalid visits")?;
        let max_pairs = args
            .next()
            .as_deref()
            .unwrap_or("10000")
            .parse()
            .map_err(|_| "invalid max pairs")?;
        let seed = args
            .next()
            .as_deref()
            .unwrap_or("42")
            .parse()
            .map_err(|_| "invalid seed")?;
        let options = Options {
            batch,
            visits,
            max_pairs,
            seed,
        };
        if let Some((model, value)) = hybrid {
            let device = args.next().unwrap_or_else(|| "cpu".into());
            if args.next().is_some() {
                return Err("too many match-hybrid arguments".into());
            }
            #[cfg(feature = "neural")]
            return chess_engine::hybrid::run(
                options,
                Path::new(&model),
                Path::new(&value),
                &device,
            )
            .map_err(|error| error.to_string());
            #[cfg(not(feature = "neural"))]
            {
                let _ = (options, model, value, device);
                return Err("build chess-engine with --features neural or cuda".into());
            }
        }
        if args.next().is_some() {
            return Err("too many match arguments".into());
        }
        return matches::run(options).map_err(|error| error.to_string());
    }
    if first == "self-play" {
        let (Some(output), Some(games), Some(seed), None) =
            (args.next(), args.next(), args.next(), args.next())
        else {
            return Err("usage: chess-engine self-play OUTPUT_DIR GAMES SEED".into());
        };
        let games = games
            .parse::<NonZeroUsize>()
            .map_err(|_| "GAMES must be a positive integer")?;
        let seed = seed
            .parse::<u64>()
            .map_err(|_| "SEED must be an unsigned 64-bit integer")?;
        let run =
            self_play::generate(Path::new(&output), games, seed).map_err(|err| err.to_string())?;
        println!("{}", run.display());
        return Ok(());
    }
    let simulations = first
        .parse::<NonZeroU32>()
        .map_err(|_| "SIMULATIONS must be a positive integer")?;
    let mut game = Game::default();

    for name in args {
        let mv = game
            .legal_moves()
            .into_iter()
            .find(|mv| mv.to_string() == name)
            .ok_or_else(|| format!("illegal move: {name}"))?;
        if !game.play(mv) {
            return Err(format!("game already ended before {name}"));
        }
    }

    if let Some(outcome) = game.outcome() {
        println!("{outcome:?}");
        return Ok(());
    }

    let mut moves = search(&game, simulations, material_value);
    moves.sort_by_key(|entry| std::cmp::Reverse(entry.visits));
    for entry in moves {
        println!("{} {}", entry.mv, entry.visits);
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}
