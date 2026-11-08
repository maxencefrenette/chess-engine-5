use std::{env, path::Path, process::ExitCode};

use chess_engine::{
    chess::Game,
    hybrid::Hybrid,
    matches::{self, Options},
    self_play,
};

const USAGE: &str = "usage: chess-engine [--cuda] <command>
  search MODEL VALUE VISITS [MOVE ...]
  self-play MODEL VALUE OUTPUT_DIR GAMES SEED [BATCH=16]
  match BASE_MODEL BASE_VALUE CANDIDATE_MODEL CANDIDATE_VALUE [BATCH=8] [VISITS=800] [MAX_PAIRS=10000] [SEED=42]";

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let first = args.next().ok_or(USAGE)?;
    let (device, command) = if first == "--cuda" {
        ("cuda", args.next().ok_or(USAGE)?)
    } else {
        ("cpu", first)
    };
    if !matches!(command.as_str(), "search" | "self-play" | "match") {
        return Err(USAGE.into());
    }
    let model = args.next().ok_or(USAGE)?;
    let value = args.next().ok_or(USAGE)?;

    match command.as_str() {
        "match" => {
            let candidate_model = args.next().ok_or(USAGE)?;
            let candidate_value = args.next().ok_or(USAGE)?;
            let options = Options {
                batch: args.next().as_deref().unwrap_or("8").parse()?,
                visits: args.next().as_deref().unwrap_or("800").parse()?,
                max_pairs: args.next().as_deref().unwrap_or("10000").parse()?,
                seed: args.next().as_deref().unwrap_or("42").parse()?,
            };
            if args.next().is_some() {
                return Err(USAGE.into());
            }
            let baseline = Hybrid::load(Path::new(&model), Path::new(&value), device)?;
            let candidate = Hybrid::load(
                Path::new(&candidate_model),
                Path::new(&candidate_value),
                device,
            )?;
            println!("baseline={model} candidate={candidate_model} device={device}");
            matches::run(options, [&baseline, &candidate])?;
        }
        "self-play" => {
            let output = args.next().ok_or(USAGE)?;
            let games = args.next().ok_or(USAGE)?.parse()?;
            let seed = args.next().ok_or(USAGE)?.parse()?;
            let batch = args.next().as_deref().unwrap_or("16").parse()?;
            if args.next().is_some() {
                return Err(USAGE.into());
            }
            let model = Hybrid::load(Path::new(&model), Path::new(&value), device)?;
            let run = self_play::generate(Path::new(&output), games, seed, batch, &model)?;
            println!("{}", run.display());
        }
        "search" => {
            let visits = args.next().ok_or(USAGE)?.parse()?;
            let mut game = Game::default();
            for name in args {
                let mv = game
                    .legal_moves()
                    .into_iter()
                    .find(|mv| mv.to_string() == name)
                    .ok_or_else(|| format!("illegal move: {name}"))?;
                if !game.play(mv) {
                    return Err(format!("game already ended before {name}").into());
                }
            }
            if let Some(outcome) = game.outcome() {
                println!("{outcome:?}");
                return Ok(());
            }
            let model = Hybrid::load(Path::new(&model), Path::new(&value), device)?;
            let mut moves = model.search(&game, visits)?;
            moves.sort_by_key(|entry| std::cmp::Reverse(entry.visits));
            for entry in moves {
                println!("{} {}", entry.mv, entry.visits);
            }
        }
        _ => unreachable!("command already checked"),
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
