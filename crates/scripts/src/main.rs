use std::{env, error::Error, fs, io::Write, path::Path, process::ExitCode};

use tempfile::NamedTempFile;
use toml_edit::{DocumentMut, value};

fn run() -> Result<(), Box<dyn Error>> {
    let mut args = env::args_os().skip(1);
    let candidate = match (args.next().as_deref(), args.next(), args.next()) {
        (Some(command), Some(candidate), None) if command == "update-model" => candidate,
        _ => return Err("usage: chess-scripts update-model CANDIDATE_DIR".into()),
    };

    let candidate = fs::canonicalize(candidate)?;
    for name in ["config.toml", "weights.safetensors"] {
        if !candidate.join(name).is_file() {
            return Err(format!("checkpoint is missing {}", candidate.join(name).display()).into());
        }
    }
    let candidate = candidate.to_str().ok_or("checkpoint path must be UTF-8")?;

    let path = Path::new("config/eval-model.toml");
    let mut config: DocumentMut = fs::read_to_string(path)?.parse()?;
    if config
        .get("baseline_dir")
        .and_then(|item| item.as_str())
        .is_none()
    {
        return Err("config/eval-model.toml must contain a baseline_dir string".into());
    }
    config["baseline_dir"] = value(candidate);

    let mut pending = NamedTempFile::new_in("config")?;
    pending
        .as_file()
        .set_permissions(fs::metadata(path)?.permissions())?;
    write!(pending, "{config}")?;
    pending.persist(path)?;
    println!("Updated baseline_dir to {candidate}");
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
