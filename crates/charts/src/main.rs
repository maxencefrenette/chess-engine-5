use chess_charts::{Result, loss_categorical, loss_numeric, nelo_categorical, nelo_numeric};
use serde::de::DeserializeOwned;
use std::{env, path::Path};

const USAGE: &str = "usage: chess-charts TYPE OUTPUT.svg X_LABEL INPUT.csv

TYPE               CSV columns
nelo-numeric       x,nelo,lower,upper
nelo-categorical   label,nelo,lower,upper
loss-numeric       x,loss
loss-categorical   label,loss

Numeric x values must be finite. Categories follow CSV order.
Supply 95% confidence bounds for nElo. For single-run total loss, charts use
approximate 95% bounds of loss ± 1.96 × loss_stddev from config/eval-model.toml.";

fn read_csv<T: DeserializeOwned>(path: &str) -> Result<Vec<T>> {
    let mut reader = csv::Reader::from_path(path)?;
    Ok(reader.deserialize().collect::<csv::Result<_>>()?)
}

fn run() -> Result<()> {
    let args: Vec<_> = env::args().skip(1).collect();
    if args.is_empty() || args == ["--help"] {
        println!("{USAGE}");
        return Ok(());
    }
    let [kind, output, x_label, input] = args.as_slice() else {
        return Err(USAGE.into());
    };
    let path = Path::new(output);
    match kind.as_str() {
        "nelo-numeric" => nelo_numeric(path, x_label, &read_csv(input)?),
        "nelo-categorical" => nelo_categorical(path, x_label, &read_csv(input)?),
        "loss-numeric" => loss_numeric(path, x_label, &read_csv(input)?),
        "loss-categorical" => loss_categorical(path, x_label, &read_csv(input)?),
        _ => Err(USAGE.into()),
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
