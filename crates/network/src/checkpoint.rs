use std::{collections::HashMap, fs, io, path::Path};

use candle_core::{DType, Device, Tensor, safetensors};
use candle_nn::VarBuilder;
use serde::{Deserialize, Serialize};

use crate::{
    ModelConfig, Net,
    piece_value::{CpuPieceValue, PREFIX, PieceValue},
};

const CONFIG_FILE: &str = "config.toml";
const METRICS_FILE: &str = "metrics.toml";
const WEIGHTS_FILE: &str = "weights.safetensors";

#[derive(Deserialize, Serialize)]
struct Config<T> {
    training: T,
}

#[derive(Deserialize)]
struct ModelSettings {
    model: ModelConfig,
}

pub struct Checkpoint {
    tensors: HashMap<String, Tensor>,
    model: ModelConfig,
}

impl Checkpoint {
    pub fn load(directory: &Path, device: &Device) -> io::Result<Self> {
        let path = directory.join(CONFIG_FILE);
        let text = fs::read_to_string(&path).map_err(|error| {
            io::Error::new(error.kind(), format!("{}: {error}", path.display()))
        })?;
        let config: Config<ModelSettings> = toml::from_str(&text)
            .map_err(|error| io::Error::other(format!("{}: {error}", path.display())))?;
        let tensors =
            safetensors::load(directory.join(WEIGHTS_FILE), device).map_err(io::Error::other)?;
        Ok(Self {
            tensors,
            model: config.training.model,
        })
    }

    fn value_tensors(&self, device: &Device) -> candle_core::Result<HashMap<String, Tensor>> {
        ["weight", "bias"]
            .into_iter()
            .map(|field| {
                let name = format!("{PREFIX}.{field}");
                let tensor = self
                    .tensors
                    .get(&name)
                    .ok_or_else(|| candle_core::Error::Msg(format!("missing tensor {name}")))?;
                Ok((name, tensor.to_device(device)?))
            })
            .collect()
    }

    pub fn value(&self) -> candle_core::Result<CpuPieceValue> {
        let device = &Device::Cpu;
        PieceValue::new(VarBuilder::from_tensors(
            self.value_tensors(device)?,
            DType::F32,
            device,
        ))?
        .into_cpu()
    }

    pub fn into_policy(self, device: &Device) -> candle_core::Result<Net> {
        Net::new(
            VarBuilder::from_tensors(self.tensors, DType::F32, device),
            self.model,
        )
    }
}

/// Save a policy and its playing value model together. Publish config last.
pub fn save(
    directory: &Path,
    tensors: HashMap<String, Tensor>,
    training: impl Serialize,
    metrics: impl Serialize,
) -> io::Result<()> {
    let config = toml::to_string_pretty(&Config { training }).map_err(io::Error::other)?;
    let metrics = toml::to_string_pretty(&metrics).map_err(io::Error::other)?;
    fs::create_dir(directory)?;
    safetensors::save(&tensors, directory.join(WEIGHTS_FILE)).map_err(io::Error::other)?;
    fs::write(directory.join(METRICS_FILE), metrics)?;
    fs::write(directory.join(CONFIG_FILE), config)
}
