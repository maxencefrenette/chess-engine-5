pub mod checkpoint;

use candle_core::{Result, Tensor};
use candle_nn::{
    Init, LayerNorm, Linear, Module, VarBuilder, layer_norm, linear, linear_no_bias,
    ops::{layer_norm_slow, softmax, softmax_last_dim},
};

pub const SQUARES: usize = 64;
pub const INPUTS: usize = 21;
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    pub width: usize,
    pub heads: usize,
    pub layers: usize,
    pub ffn_width: usize,
    pub smolgen_compression: usize,
    pub smolgen_hidden: usize,
    pub smolgen_head_width: usize,
    pub policy_width: usize,
    pub value_channels: usize,
    pub value_hidden_width: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        // Dimensions of the original v1 checkpoints, before they saved all dimensions.
        Self {
            width: 64,
            heads: 4,
            layers: 2,
            ffn_width: 128,
            smolgen_compression: 2,
            smolgen_hidden: 64,
            smolgen_head_width: 8,
            policy_width: 32,
            value_channels: 4,
            value_hidden_width: 64,
        }
    }
}

impl ModelConfig {
    fn validate(self) -> Result<()> {
        if [
            self.width,
            self.heads,
            self.layers,
            self.ffn_width,
            self.smolgen_compression,
            self.smolgen_hidden,
            self.smolgen_head_width,
            self.policy_width,
            self.value_channels,
            self.value_hidden_width,
        ]
        .contains(&0)
            || !self.width.is_multiple_of(self.heads)
        {
            candle_core::bail!(
                "model dimensions must be positive and width must be divisible by heads"
            );
        }
        Ok(())
    }
}

fn normalize(norm: &LayerNorm, x: &Tensor) -> Result<Tensor> {
    if !x.track_op() && !norm.weight().track_op() {
        return norm.forward(x);
    }
    // Candle's fused LayerNorm uses a forward-only custom op.
    layer_norm_slow(
        x,
        norm.weight(),
        norm.bias().expect("affine layer norm"),
        norm.eps() as f32,
    )
}

struct Smolgen {
    compress: Linear,
    mix: Linear,
    mix_norm: LayerNorm,
    heads: Linear,
    head_norm: LayerNorm,
}

impl Smolgen {
    fn new(vb: VarBuilder, config: ModelConfig) -> Result<Self> {
        Ok(Self {
            compress: linear_no_bias(config.width, config.smolgen_compression, vb.pp("compress"))?,
            mix: linear(
                SQUARES * config.smolgen_compression,
                config.smolgen_hidden,
                vb.pp("mix"),
            )?,
            mix_norm: layer_norm(config.smolgen_hidden, 1e-5, vb.pp("mix_norm"))?,
            heads: linear(
                config.smolgen_hidden,
                config.heads * config.smolgen_head_width,
                vb.pp("heads"),
            )?,
            head_norm: layer_norm(
                config.heads * config.smolgen_head_width,
                1e-5,
                vb.pp("head_norm"),
            )?,
        })
    }

    fn forward(&self, x: &Tensor, projection: &Linear, config: ModelConfig) -> Result<Tensor> {
        let batch = x.dim(0)?;
        let compressed = self.compress.forward(x)?.flatten_from(1)?;
        let mixed = normalize(&self.mix_norm, &self.mix.forward(&compressed)?.silu()?)?;
        let heads = normalize(&self.head_norm, &self.heads.forward(&mixed)?.silu()?)?;
        projection
            .forward(&heads.reshape((batch, config.heads, config.smolgen_head_width))?)?
            .reshape((batch, config.heads, SQUARES, SQUARES))
    }
}

struct Block {
    qkv: Linear,
    attention_out: Linear,
    attention_norm: LayerNorm,
    expand: Linear,
    contract: Linear,
    ffn_norm: LayerNorm,
    smolgen: Smolgen,
}

impl Block {
    fn new(vb: VarBuilder, config: ModelConfig) -> Result<Self> {
        Ok(Self {
            qkv: linear(config.width, 3 * config.width, vb.pp("qkv"))?,
            attention_out: linear(config.width, config.width, vb.pp("attention_out"))?,
            attention_norm: layer_norm(config.width, 1e-5, vb.pp("attention_norm"))?,
            expand: linear(config.width, config.ffn_width, vb.pp("expand"))?,
            contract: linear(config.ffn_width, config.width, vb.pp("contract"))?,
            ffn_norm: layer_norm(config.width, 1e-5, vb.pp("ffn_norm"))?,
            smolgen: Smolgen::new(vb.pp("smolgen"), config)?,
        })
    }

    fn forward(&self, x: &Tensor, projection: &Linear, config: ModelConfig) -> Result<Tensor> {
        let batch = x.dim(0)?;
        let qkv = self.qkv.forward(x)?;
        let heads = |part| -> Result<Tensor> {
            qkv.narrow(2, part * config.width, config.width)?
                .reshape((batch, SQUARES, config.heads, config.width / config.heads))?
                .transpose(1, 2)?
                .contiguous()
        };
        let (q, k, v) = (heads(0)?, heads(1)?, heads(2)?);
        let scores = q
            .matmul(&k.transpose(2, 3)?)?
            .affine(((config.width / config.heads) as f64).sqrt().recip(), 0.)?;
        let bias = self.smolgen.forward(x, projection, config)?;
        let scores = (&scores + &bias)?;
        let probs = if scores.track_op() {
            softmax(&scores, 3)?
        } else {
            softmax_last_dim(&scores)?
        };
        let attention =
            probs
                .matmul(&v)?
                .transpose(1, 2)?
                .reshape((batch, SQUARES, config.width))?;
        let x = normalize(
            &self.attention_norm,
            &(x + self.attention_out.forward(&attention)?)?,
        )?;
        let ff = self
            .contract
            .forward(&self.expand.forward(&x)?.relu()?.sqr()?)?;
        normalize(&self.ffn_norm, &(&x + &ff)?)
    }
}

pub struct Net {
    config: ModelConfig,
    input: Linear,
    square: Tensor,
    layers: Vec<Block>,
    smolgen_projection: Linear,
    policy_query: Linear,
    policy_key: Linear,
    promotion: Linear,
    value_square: Linear,
    value_hidden: Linear,
    value_out: Linear,
    material_hint: Option<Linear>,
}

impl Net {
    pub fn new(vb: VarBuilder, config: ModelConfig, material_hint: bool) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            input: linear(INPUTS, config.width, vb.pp("input"))?,
            square: vb.get_with_hints(
                (SQUARES, config.width),
                "square",
                Init::Randn {
                    mean: 0.0,
                    stdev: 0.02,
                },
            )?,
            layers: (0..config.layers)
                .map(|i| Block::new(vb.pp("blocks").pp(i), config))
                .collect::<Result<_>>()?,
            smolgen_projection: linear_no_bias(
                config.smolgen_head_width,
                SQUARES * SQUARES,
                vb.pp("smolgen_projection"),
            )?,
            policy_query: linear(config.width, config.policy_width, vb.pp("policy_query"))?,
            policy_key: linear(config.width, config.policy_width, vb.pp("policy_key"))?,
            promotion: linear_no_bias(config.policy_width, 4, vb.pp("promotion"))?,
            value_square: linear(config.width, config.value_channels, vb.pp("value_square"))?,
            value_hidden: linear(
                SQUARES * config.value_channels,
                config.value_hidden_width,
                vb.pp("value_hidden"),
            )?,
            value_out: linear(config.value_hidden_width, 3, vb.pp("value_out"))?,
            // Start the residual at zero so the hinted and plain nets have the same initial function.
            material_hint: material_hint
                .then(|| {
                    vb.get_with_hints(
                        (config.value_hidden_width, 1),
                        "material_hint.weight",
                        Init::Const(0.0),
                    )
                    .map(|weight| Linear::new(weight, None))
                })
                .transpose()?,
        })
    }

    pub fn forward(
        &self,
        position: &Tensor,
        move_index: &Tensor,
        promotion_index: &Tensor,
        promotes: &Tensor,
        material: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let x = self.encode(position)?;
        let policy = self.legal_policy(&x, move_index, promotion_index, promotes)?;
        let value = self.value_square.forward(&x)?.relu()?.flatten_from(1)?;
        let mut value = self.value_hidden.forward(&value)?;
        if let Some(projection) = &self.material_hint {
            let material =
                material.ok_or_else(|| candle_core::Error::Msg("missing material hint".into()))?;
            value = (&value + projection.forward(material)?)?;
        }
        let value = self.value_out.forward(&value.relu()?)?;
        Ok((policy, value))
    }

    pub fn policy(
        &self,
        position: &Tensor,
        move_index: &Tensor,
        promotion_index: &Tensor,
        promotes: &Tensor,
    ) -> Result<Tensor> {
        self.legal_policy(
            &self.encode(position)?,
            move_index,
            promotion_index,
            promotes,
        )
    }

    fn encode(&self, position: &Tensor) -> Result<Tensor> {
        let mut x = self.input.forward(position)?.broadcast_add(&self.square)?;
        for layer in &self.layers {
            x = layer.forward(&x, &self.smolgen_projection, self.config)?;
        }
        Ok(x)
    }

    fn legal_policy(
        &self,
        x: &Tensor,
        move_index: &Tensor,
        promotion_index: &Tensor,
        promotes: &Tensor,
    ) -> Result<Tensor> {
        let q = self.policy_query.forward(x)?;
        let k = self.policy_key.forward(x)?;
        let logits = q
            .matmul(&k.transpose(1, 2)?)?
            .affine((self.config.policy_width as f64).sqrt().recip(), 0.)?;
        let legal = logits.flatten_from(1)?.gather(move_index, 1)?;
        let offsets = self
            .promotion
            .forward(&k)?
            .flatten_from(1)?
            .gather(promotion_index, 1)?;
        &legal + (&offsets * promotes)?
    }
}

/// Append 64 square tokens from the v1 packed-board representation.
pub fn encode_position(
    output: &mut Vec<f32>,
    board: &[u8; 32],
    black_to_move: bool,
    castling: u8,
    ep_square: u8,
    halfmove_clock: u8,
    repetitions: u8,
) -> Option<()> {
    for square in 0..SQUARES {
        let piece = (board[square / 2] >> (4 * (square % 2))) & 15;
        if piece > 12 {
            return None;
        }
        let mut token = [0.; INPUTS];
        token[piece as usize] = 1.;
        token[13] = f32::from(black_to_move);
        for bit in 0..4 {
            token[14 + bit] = f32::from((castling >> bit) & 1);
        }
        token[18] = f32::from(ep_square as usize == square);
        token[19] = f32::from(halfmove_clock) / 100.;
        token[20] = f32::from(repetitions) - 1.;
        output.extend(token);
    }
    Some(())
}

pub fn move_indices(mv: u16) -> (u32, u32, f32) {
    let from = u32::from(mv & 63);
    let to = u32::from((mv >> 6) & 63);
    let promotion = u32::from(mv >> 12);
    (
        from * 64 + to,
        to * 4 + promotion.saturating_sub(1),
        f32::from(promotion != 0),
    )
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    use super::*;

    #[test]
    fn value_loss_reaches_smolgen_and_material_hint() -> Result<()> {
        let device = &Device::Cpu;
        let params = VarMap::new();
        let net = Net::new(
            VarBuilder::from_varmap(&params, DType::F32, device),
            ModelConfig::default(),
            true,
        )?;
        let board = Tensor::randn(0f32, 1., (1, SQUARES, INPUTS), device)?;
        let moves = Tensor::zeros((1, 1), DType::U32, device)?;
        let promotes = Tensor::zeros((1, 1), DType::F32, device)?;
        let material = Tensor::ones((1, 1), DType::F32, device)?;
        let (_, value) = net.forward(&board, &moves, &moves, &promotes, Some(&material))?;
        let grads = value.sum_all()?.backward()?;
        let params = params.data().lock().unwrap();
        for name in [
            "input.weight",
            "blocks.0.smolgen.compress.weight",
            "smolgen_projection.weight",
            "material_hint.weight",
        ] {
            let grad = grads
                .get(params[name].as_tensor())
                .unwrap_or_else(|| panic!("no gradient for {name}"));
            assert!(grad.abs()?.sum_all()?.to_scalar::<f32>()? > 0.0, "{name}");
        }
        Ok(())
    }
}
