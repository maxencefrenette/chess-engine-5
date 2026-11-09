use std::{error::Error, fs};

use serde::de::DeserializeOwned;
use toml::{Table, Value};

pub fn load<T: DeserializeOwned>(
    default: &str,
    mut args: impl Iterator<Item = String>,
) -> Result<T, Box<dyn Error>> {
    let mut path = None;
    let mut overrides = Vec::new();
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--config" if path.is_none() => {
                path = Some(args.next().ok_or("--config needs a path")?)
            }
            "--set" => overrides.push(args.next().ok_or("--set needs KEY=VALUE")?),
            _ => return Err(format!("unexpected argument {flag}").into()),
        }
    }
    let path = path.as_deref().unwrap_or(default);
    let text = fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
    let mut config: Value = toml::from_str(&text).map_err(|error| format!("{path}: {error}"))?;
    for entry in overrides {
        let (key, raw) = entry.split_once('=').ok_or("--set needs KEY=VALUE")?;
        let target = key
            .split('.')
            .try_fold(&mut config, |node, part| node.get_mut(part))
            .ok_or_else(|| format!("unknown config key {key}"))?;
        *target = if target.is_str() {
            Value::String(raw.into())
        } else {
            let mut parsed: Table = toml::from_str(&format!("value = {raw}"))
                .map_err(|error| format!("invalid override for {key}: {error}"))?;
            if parsed.len() != 1 {
                return Err(format!("override for {key} must contain one TOML value").into());
            }
            parsed.remove("value").ok_or("missing override value")?
        };
    }
    Ok(config.try_into()?)
}
