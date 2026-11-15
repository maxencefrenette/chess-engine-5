use std::{env, path::Path, process::ExitCode};

use chess_config::{DeviceConfig, TomlConfig};
use chess_engine::{
    chess::Game,
    matches::{self, Mode, Options},
    model_evaluator::ModelEvaluator,
    self_play,
    settings::Settings,
};

const USAGE: &str = "usage: chess-engine [--cuda] <command>
  search MODEL_DIR VISITS [MOVE ...]
  self-play [--config PATH] [--set KEY=VALUE]...
  eval-model <gainer|non-regression|fixed> CANDIDATE_DIR [--matched-speed] [--config PATH] [--set KEY=VALUE]...";

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let first = args.next().ok_or(USAGE)?;
    let (device, command) = if first == "--cuda" {
        (DeviceConfig::Cuda, args.next().ok_or(USAGE)?)
    } else {
        (DeviceConfig::Cpu, first)
    };
    if command == "eval-model" && device == DeviceConfig::Cpu {
        let mode = match args.next().as_deref() {
            Some("gainer") => Mode::Gainer,
            Some("non-regression") => Mode::NonRegression,
            Some("fixed") => Mode::Fixed,
            _ => return Err(USAGE.into()),
        };
        let candidate_dir = args
            .next()
            .filter(|arg| !arg.starts_with("--"))
            .ok_or(USAGE)?;
        let (options, matched_speed) = eval_options(args)?;
        let settings = Settings::load()?;
        let baseline = ModelEvaluator::load(&options.baseline_dir, options.device)?;
        let candidate = ModelEvaluator::load(Path::new(&candidate_dir), options.device)?;
        println!(
            "baseline={} candidate={} device={}",
            options.baseline_dir.display(),
            candidate_dir,
            options.device
        );
        return Ok(matches::run(
            options,
            mode,
            matched_speed,
            [&baseline, &candidate],
            &settings,
        )?);
    }
    if command == "self-play" && device == DeviceConfig::Cpu {
        let config = self_play::Config::load("config/datagen.toml", args)?;
        let model = ModelEvaluator::load(&config.model, config.device)?;
        let run = self_play::generate(&config, &model, &Settings::load()?)?;
        println!("{}", run.display());
        return Ok(());
    }
    match command.as_str() {
        "search" => {
            let model = args.next().ok_or(USAGE)?;
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
            let model = ModelEvaluator::load(Path::new(&model), device)?;
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

fn eval_options(
    mut args: impl Iterator<Item = String>,
) -> Result<(Options, bool), Box<dyn std::error::Error>> {
    let mut matched_speed = false;
    let mut config_args = Vec::new();
    while let Some(arg) = args.next() {
        if arg == "--matched-speed" {
            if matched_speed {
                return Err("--matched-speed specified twice".into());
            }
            matched_speed = true;
        } else {
            let takes_value = matches!(arg.as_str(), "--config" | "--set");
            config_args.push(arg);
            if takes_value {
                config_args.push(args.next().ok_or("missing config argument value")?);
            }
        }
    }
    Ok((
        Options::load("config/eval-model.toml", config_args.into_iter())?,
        matched_speed,
    ))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matched_speed_flag_works_alongside_config_overrides() {
        let config_args = [
            "--config",
            concat!(env!("CARGO_MANIFEST_DIR"), "/../../config/eval-model.toml"),
        ];
        for args in [
            vec!["--matched-speed", "--set", "baseline_visits_per_move=123"],
            vec!["--set", "baseline_visits_per_move=123", "--matched-speed"],
        ] {
            let (_, matched_speed) =
                eval_options(config_args.into_iter().chain(args).map(String::from)).unwrap();
            assert!(matched_speed);
        }
        assert!(
            !eval_options(config_args.into_iter().map(String::from))
                .unwrap()
                .1
        );
        assert!(
            eval_options(
                ["--matched-speed", "--matched-speed"]
                    .into_iter()
                    .map(String::from)
            )
            .is_err()
        );
        assert!(eval_options(["--set", "--matched-speed"].into_iter().map(String::from)).is_err());
    }
}
