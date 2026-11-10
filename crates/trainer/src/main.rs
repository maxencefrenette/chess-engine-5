use std::{env, error::Error, process::ExitCode};

use chess_config::TomlConfig;

mod data;
mod train;

fn run() -> Result<(), Box<dyn Error>> {
    let config = train::Config::load("config/training.toml", env::args().skip(1))?;
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
