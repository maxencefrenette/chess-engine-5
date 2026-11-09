use std::{env, path::Path, process::ExitCode};

use chess_engine::{
    chess::Game,
    hybrid::Hybrid,
    matches::{self, Mode, Options},
    self_play,
    settings::Settings,
};

const USAGE: &str = "usage: chess-engine [--cuda] <command>
  search MODEL_DIR VISITS [MOVE ...]
  self-play MODEL_DIR OUTPUT_DIR GAMES SEED [BATCH=16]
  eval-model <gainer|non-regression> CANDIDATE_DIR [--config PATH] [--set KEY=VALUE]...";

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let first = args.next().ok_or(USAGE)?;
    let (device, command) = if first == "--cuda" {
        ("cuda", args.next().ok_or(USAGE)?)
    } else {
        ("cpu", first)
    };
    if command == "eval-model" && device == "cpu" {
        let mode = match args.next().as_deref() {
            Some("gainer") => Mode::Gainer,
            Some("non-regression") => Mode::NonRegression,
            _ => return Err(USAGE.into()),
        };
        let candidate_dir = args
            .next()
            .filter(|arg| !arg.starts_with("--"))
            .ok_or(USAGE)?;
        let options: Options = chess_config::load("config/eval-model.toml", args)?;
        let settings = Settings::load()?;
        let baseline = Hybrid::load(&options.baseline_dir, &options.device)?;
        let candidate = Hybrid::load(Path::new(&candidate_dir), &options.device)?;
        println!(
            "baseline={} candidate={} device={}",
            options.baseline_dir.display(),
            candidate_dir,
            options.device
        );
        return Ok(matches::run(
            options,
            mode,
            [&baseline, &candidate],
            &settings,
        )?);
    }
    let model = args.next().ok_or(USAGE)?;

    match command.as_str() {
        "self-play" => {
            let output = args.next().ok_or(USAGE)?;
            let games = args.next().ok_or(USAGE)?.parse()?;
            let seed = args.next().ok_or(USAGE)?.parse()?;
            let batch = args.next().as_deref().unwrap_or("16").parse()?;
            if args.next().is_some() {
                return Err(USAGE.into());
            }
            let model = Hybrid::load(Path::new(&model), device)?;
            let settings = Settings::load()?;
            let run =
                self_play::generate(Path::new(&output), games, seed, batch, &model, &settings)?;
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
            let model = Hybrid::load(Path::new(&model), device)?;
            let mut moves = model.search(&game, visits, Settings::load()?.exploration)?;
            moves.sort_by_key(|entry| std::cmp::Reverse(entry.visits));
            for entry in moves {
                println!("{} {}", entry.mv, entry.visits);
            }
        }
        _ => return Err(USAGE.into()),
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
