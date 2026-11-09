use std::{env, error::Error, process::ExitCode};

use candle_core::{DType, Device, Tensor};
use candle_nn::{Optimizer, SGD, VarBuilder, VarMap};
use chess_config::{DeviceConfig, TomlConfig};

mod data;
mod train;

const USAGE: &str =
    "usage: chess-trainer check-device [cpu|cuda] | train [--config PATH] [--set KEY=VALUE]...";

fn run() -> Result<(), Box<dyn Error>> {
    let mut args = env::args().skip(1);
    let command = args.next();
    match command.as_deref() {
        Some("check-device") => {
            let backend = args.next().as_deref().unwrap_or("cpu").parse()?;
            if args.next().is_some() {
                return Err(USAGE.into());
            }
            check_device(backend, &backend.try_into()?)
        }
        Some("train") => {
            let config = train::Config::load("config/training.toml", args)?;
            let device = config.device.try_into()?;
            train::run(config, &device)
        }
        _ => Err(USAGE.into()),
    }
}

fn check_device(backend: DeviceConfig, device: &Device) -> Result<(), Box<dyn Error>> {
    // Verify a gradient and parameter update on the selected device.
    let parameters = VarMap::new();
    let builder = VarBuilder::from_varmap(&parameters, DType::F32, device);
    let weight = builder.get_with_hints((), "weight", candle_nn::Init::Const(0.0))?;
    let target = Tensor::new(1.0_f32, device)?;
    let loss = || (&weight - &target)?.sqr();
    let before = loss()?.to_scalar::<f32>()?;
    let mut optimizer = SGD::new(parameters.all_vars(), 0.1)?;
    optimizer.backward_step(&loss()?)?;
    let after = loss()?.to_scalar::<f32>()?;
    if !after.is_finite() || after >= before {
        return Err(
            format!("Candle optimizer step did not reduce loss: {before} -> {after}").into(),
        );
    }

    println!("Candle {backend} check passed: loss {before:.3} -> {after:.3}");
    Ok(())
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
