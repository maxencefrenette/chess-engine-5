use std::{error::Error, fs};

pub use chess_config_derive::{TomlConfig, toml_config};
use serde::de::DeserializeOwned;
use toml::{Table, Value};

pub trait TomlConfig: DeserializeOwned {
    fn from_repo_file(name: &str) -> Result<Self, Box<dyn Error>> {
        Self::from_file(&format!(
            "{}/../../config/{name}",
            env!("CARGO_MANIFEST_DIR")
        ))
    }

    fn from_file(path: &str) -> Result<Self, Box<dyn Error>> {
        Self::load(path, std::iter::empty())
    }

    fn load(default: &str, mut args: impl Iterator<Item = String>) -> Result<Self, Box<dyn Error>> {
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
        let mut config: Value =
            toml::from_str(&text).map_err(|error| format!("{path}: {error}"))?;
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
        config
            .try_into()
            .map_err(|error| format!("{path}: {error}").into())
    }
}

#[derive(Clone, Copy, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum DeviceConfig {
    Cpu,
    Cuda,
}

impl TryFrom<DeviceConfig> for candle_core::Device {
    type Error = candle_core::Error;

    fn try_from(config: DeviceConfig) -> Result<Self, Self::Error> {
        match config {
            DeviceConfig::Cpu => Ok(Self::Cpu),
            DeviceConfig::Cuda => Self::new_cuda(0),
        }
    }
}

impl std::fmt::Display for DeviceConfig {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
        })
    }
}

impl std::str::FromStr for DeviceConfig {
    type Err = &'static str;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "cpu" => Ok(Self::Cpu),
            "cuda" => Ok(Self::Cuda),
            _ => Err("device must be cpu or cuda"),
        }
    }
}
