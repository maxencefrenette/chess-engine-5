use std::{env, error::Error, process::ExitCode};

use chess_config::TomlConfig;

mod data;
mod train;

const USAGE: &str = "usage: chess-trainer train [--config PATH] [--set KEY=VALUE]...";

fn run() -> Result<(), Box<dyn Error>> {
    let mut args = env::args().skip(1);
    if args.next().as_deref() != Some("train") {
        return Err(USAGE.into());
    }
    let config = train::Config::load("config/training.toml", args)?;
    let device = config.device.try_into()?;
    train::run(config, &device)
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}
