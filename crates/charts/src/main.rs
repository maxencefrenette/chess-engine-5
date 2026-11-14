use std::{env, path::Path};

use chess_charts::{Estimate, Result, Series, SweepPoint};

const USAGE: &str = "usage: chess-charts
  training OUTPUT.svg METRIC LABEL=training.csv [LABEL=training.csv ...]
  nelo OUTPUT.svg INPUT.csv
  sweep OUTPUT.svg X_LABEL INPUT.csv

Training curves use elapsed_time (seconds) on the x-axis.
nElo CSV columns: label,nelo,lower,upper (95% confidence bounds).
Sweep CSV columns: x,nelo,lower,upper (95% confidence bounds).";

fn column(headers: &csv::StringRecord, name: &str) -> Result<usize> {
    headers
        .iter()
        .position(|h| h == name)
        .ok_or_else(|| format!("missing CSV column: {name}").into())
}

fn run() -> Result<()> {
    let args: Vec<_> = env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        None | Some("--help") => {
            println!("{USAGE}");
            Ok(())
        }
        Some("training") if args.len() >= 4 => {
            let mut series = Vec::new();
            for input in &args[3..] {
                let (label, path) = input.split_once('=').ok_or("expected LABEL=CSV")?;
                let mut reader = csv::Reader::from_path(path)?;
                let x = column(reader.headers()?, "elapsed_time")?;
                let y = column(reader.headers()?, &args[2])?;
                let points = reader
                    .records()
                    .map(|row| {
                        let row = row?;
                        Ok((row[x].parse()?, row[y].parse()?))
                    })
                    .collect::<Result<Vec<_>>>()?;
                series.push(Series {
                    label: label.into(),
                    points,
                });
            }
            chess_charts::lines(
                Path::new(&args[1]),
                &format!("Training: {}", args[2]),
                "Elapsed training time (s)",
                &args[2],
                &series,
            )
        }
        Some("nelo") if args.len() == 3 => {
            let mut reader = csv::Reader::from_path(&args[2])?;
            let indices = ["label", "nelo", "lower", "upper"]
                .map(|name| column(reader.headers()?, name))
                .into_iter()
                .collect::<Result<Vec<_>>>()?;
            let estimates = reader
                .records()
                .map(|row| {
                    let row = row?;
                    Ok(Estimate {
                        label: row[indices[0]].into(),
                        nelo: row[indices[1]].parse()?,
                        lower: row[indices[2]].parse()?,
                        upper: row[indices[3]].parse()?,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            chess_charts::nelo(Path::new(&args[1]), "Model comparison", &estimates)
        }
        Some("sweep") if args.len() == 4 => {
            let mut reader = csv::Reader::from_path(&args[3])?;
            let indices = ["x", "nelo", "lower", "upper"]
                .map(|name| column(reader.headers()?, name))
                .into_iter()
                .collect::<Result<Vec<_>>>()?;
            let points = reader
                .records()
                .map(|row| {
                    let row = row?;
                    Ok(SweepPoint {
                        x: row[indices[0]].parse()?,
                        nelo: row[indices[1]].parse()?,
                        lower: row[indices[2]].parse()?,
                        upper: row[indices[3]].parse()?,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            chess_charts::sweep(
                Path::new(&args[1]),
                &format!("nElo vs {}", args[2]),
                &args[2],
                &points,
            )
        }
        _ => Err(USAGE.into()),
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
