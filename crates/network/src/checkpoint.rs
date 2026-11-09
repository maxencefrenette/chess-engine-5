use std::{collections::HashMap, fs, io, path::Path};

use candle_core::{DType, Device, Tensor, safetensors};
use candle_nn::VarBuilder;
use serde_json::{Value, json};

use crate::{
    INPUTS, ModelConfig, Net,
    piece_value::{CpuPieceValue, PREFIX, PieceValue},
};

const CONFIG_FILE: &str = "config.json";
const WEIGHTS_FILE: &str = "weights.safetensors";

fn format_fields() -> [(&'static str, Value); 6] {
    [
        ("format_version", json!(1)),
        ("architecture", json!("lc0_style_smolgen_v1")),
        ("position_encoding", json!("packed_board_planes_v1")),
        ("move_encoding", json!("ce5_packed_move_v1")),
        ("value_model", json!("side_relative_piece_counts_v1")),
        ("input_channels", json!(INPUTS)),
    ]
}

pub fn config(model: ModelConfig, training: Value) -> Value {
    let mut config = json!({
        "value_order": ["loss", "draw", "win"],
        "training": training,
        "model": model,
    });
    for (key, expected) in format_fields() {
        config[key] = expected;
    }
    config
}

pub struct Checkpoint {
    tensors: HashMap<String, Tensor>,
    model: ModelConfig,
}

impl Checkpoint {
    pub fn load(directory: &Path, device: &Device) -> io::Result<Self> {
        let path = directory.join(CONFIG_FILE);
        let config: Value = serde_json::from_slice(&fs::read(&path).map_err(|error| {
            io::Error::new(error.kind(), format!("{}: {error}", path.display()))
        })?)?;
        for (key, expected) in format_fields() {
            if config[key] != expected {
                return Err(io::Error::other(format!(
                    "unsupported checkpoint {key}: {}",
                    config[key]
                )));
            }
        }
        if config["value_order"] != json!(["loss", "draw", "win"]) {
            return Err(io::Error::other("unsupported checkpoint value order"));
        }
        let model = config
            .get("model")
            .ok_or_else(|| io::Error::other("checkpoint is missing its model section"))?;
        let model = serde_json::from_value(model.clone())?;
        let tensors =
            safetensors::load(directory.join(WEIGHTS_FILE), device).map_err(io::Error::other)?;
        Ok(Self { tensors, model })
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
pub fn save(directory: &Path, tensors: HashMap<String, Tensor>, config: &Value) -> io::Result<()> {
    fs::create_dir(directory)?;
    safetensors::save(&tensors, directory.join(WEIGHTS_FILE)).map_err(io::Error::other)?;
    fs::write(directory.join(CONFIG_FILE), format!("{config:#}\n"))
}
