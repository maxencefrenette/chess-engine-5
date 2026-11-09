use std::{collections::HashMap, fs, io, path::Path};

use candle_core::{DType, Device, Tensor, safetensors};
use candle_nn::VarBuilder;
use serde_json::{Value, json};

use crate::{BLOCKS, HEADS, INPUTS, Net, WIDTH};

const CONFIG_FILE: &str = "config.json";
const WEIGHTS_FILE: &str = "weights.safetensors";
const VALUE_WEIGHT: &str = "piece_count_value.weight";
const VALUE_BIAS: &str = "piece_count_value.bias";
const MATERIAL_HINT: &str = "stm_material_squash_8_value_residual_v1";

pub struct PieceValue {
    pub weight: [[f32; 10]; 3],
    pub bias: [f32; 3],
}

fn format_fields() -> [(&'static str, Value); 9] {
    [
        ("format_version", json!(1)),
        ("architecture", json!("lc0_style_smolgen_v1")),
        ("position_encoding", json!("packed_board_planes_v1")),
        ("move_encoding", json!("ce5_packed_move_v1")),
        ("value_model", json!("side_relative_piece_counts_v1")),
        ("width", json!(WIDTH)),
        ("heads", json!(HEADS)),
        ("blocks", json!(BLOCKS)),
        ("input_channels", json!(INPUTS)),
    ]
}

pub fn config(material_hint: bool, training: Value) -> Value {
    let mut config = json!({
        "material_hint": material_hint.then_some(MATERIAL_HINT),
        "value_order": ["loss", "draw", "win"],
        "training": training,
    });
    for (key, expected) in format_fields() {
        config[key] = expected;
    }
    config
}

pub struct Checkpoint {
    tensors: HashMap<String, Tensor>,
    material_hint: bool,
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
        let material_hint = match &config["material_hint"] {
            Value::Null => false,
            hint if hint == MATERIAL_HINT => true,
            _ => return Err(io::Error::other("unsupported material hint")),
        };
        let tensors =
            safetensors::load(directory.join(WEIGHTS_FILE), device).map_err(io::Error::other)?;
        Ok(Self {
            tensors,
            material_hint,
        })
    }

    pub fn value(&self) -> io::Result<PieceValue> {
        let vb = |name| {
            self.tensors
                .get(name)
                .ok_or_else(|| io::Error::other(format!("missing tensor {name}")))
        };
        let weight = vb(VALUE_WEIGHT)?
            .to_vec2::<f32>()
            .map_err(io::Error::other)?;
        let bias = vb(VALUE_BIAS)?.to_vec1::<f32>().map_err(io::Error::other)?;
        let [loss, draw, win] = weight
            .try_into()
            .map_err(|_| io::Error::other("invalid piece-count weight shape"))?;
        let row = |values: Vec<f32>| {
            values
                .try_into()
                .map_err(|_| io::Error::other("invalid piece-count weight shape"))
        };
        let weight = [row(loss)?, row(draw)?, row(win)?];
        let bias = bias
            .try_into()
            .map_err(|_| io::Error::other("invalid piece-count bias shape"))?;
        Ok(PieceValue { weight, bias })
    }

    pub fn into_policy(self, device: &Device) -> candle_core::Result<Net> {
        Net::new(
            VarBuilder::from_tensors(self.tensors, DType::F32, device),
            self.material_hint,
        )
    }
}

/// Save a policy and its playing value model together. Publish config last.
pub fn save(
    directory: &Path,
    mut tensors: HashMap<String, Tensor>,
    value: &PieceValue,
    config: &Value,
) -> io::Result<()> {
    let device = &Device::Cpu;
    let weight = Tensor::new(&value.weight, device).map_err(io::Error::other)?;
    let bias = Tensor::new(&value.bias, device).map_err(io::Error::other)?;
    tensors.insert(VALUE_WEIGHT.into(), weight);
    tensors.insert(VALUE_BIAS.into(), bias);
    fs::create_dir(directory)?;
    safetensors::save(&tensors, directory.join(WEIGHTS_FILE)).map_err(io::Error::other)?;
    fs::write(directory.join(CONFIG_FILE), format!("{config:#}\n"))
}
