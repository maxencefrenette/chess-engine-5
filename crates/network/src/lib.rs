pub mod checkpoint;
pub mod inference;
pub mod policy_inputs;
pub mod value;

use candle_core::{Result, Tensor};
use candle_nn::{
    Init, LayerNorm, Linear, Module, VarBuilder, layer_norm, linear, linear_no_bias,
    ops::{layer_norm_slow, softmax, softmax_last_dim},
};
use value::{CpuPieceValue, ValueConfig, ValueHead};

pub const SQUARES: usize = 64;
pub const INPUTS: usize = 21;
#[chess_config::toml_config]
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct ModelConfig {
    pub width: usize,
    pub heads: usize,
    pub layers: usize,
    pub ffn_width: usize,
    pub policy_width: usize,
    pub smolgen: SmolgenConfig,
    pub value: ValueConfig,
}

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
pub enum SmolgenConfig {
    Enabled {
        compression: usize,
        hidden: usize,
        head_width: usize,
    },
    Disabled {},
}

impl ModelConfig {
    fn validate(self) -> Result<()> {
        if [
            self.width,
            self.heads,
            self.layers,
            self.ffn_width,
            self.policy_width,
        ]
        .contains(&0)
            || !self.width.is_multiple_of(self.heads)
        {
            candle_core::bail!(
                "model dimensions must be positive and width must be divisible by heads"
            );
        }
        if let SmolgenConfig::Enabled {
            compression,
            hidden,
            head_width,
        } = self.smolgen
            && [compression, hidden, head_width].contains(&0)
        {
            candle_core::bail!("Smolgen dimensions must be positive");
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
    head_width: usize,
    compress: Linear,
    mix: Linear,
    mix_norm: LayerNorm,
    heads: Linear,
    head_norm: LayerNorm,
}

impl Smolgen {
    fn new(
        vb: VarBuilder,
        config: ModelConfig,
        compression: usize,
        hidden: usize,
        head_width: usize,
    ) -> Result<Self> {
        let heads_width = config.heads * head_width;
        Ok(Self {
            head_width,
            compress: linear_no_bias(config.width, compression, vb.pp("compress"))?,
            mix: linear(SQUARES * compression, hidden, vb.pp("mix"))?,
            mix_norm: layer_norm(hidden, 1e-5, vb.pp("mix_norm"))?,
            heads: linear(hidden, heads_width, vb.pp("heads"))?,
            head_norm: layer_norm(heads_width, 1e-5, vb.pp("head_norm"))?,
        })
    }

    fn forward(&self, x: &Tensor, projection: &Linear, config: ModelConfig) -> Result<Tensor> {
        let batch = x.dim(0)?;
        let compressed = self.compress.forward(x)?.flatten_from(1)?;
        let mixed = normalize(&self.mix_norm, &self.mix.forward(&compressed)?.silu()?)?;
        let heads = normalize(&self.head_norm, &self.heads.forward(&mixed)?.silu()?)?;
        projection
            .forward(&heads.reshape((batch, config.heads, self.head_width))?)?
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
    smolgen: Option<Smolgen>,
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
            smolgen: match config.smolgen {
                SmolgenConfig::Enabled {
                    compression,
                    hidden,
                    head_width,
                } => Some(Smolgen::new(
                    vb.pp("smolgen"),
                    config,
                    compression,
                    hidden,
                    head_width,
                )?),
                SmolgenConfig::Disabled {} => None,
            },
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        projection: Option<&Linear>,
        config: ModelConfig,
    ) -> Result<Tensor> {
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
        let scores = if let Some(smolgen) = &self.smolgen {
            (&scores + &smolgen.forward(x, projection.expect("Smolgen projection"), config)?)?
        } else {
            scores
        };
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
    smolgen_projection: Option<Linear>,
    policy_query: Linear,
    policy_key: Linear,
    promotion: Linear,
    value: ValueHead,
}

impl Net {
    pub fn new(vb: VarBuilder, config: ModelConfig) -> Result<Self> {
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
            smolgen_projection: match config.smolgen {
                SmolgenConfig::Enabled { head_width, .. } => Some(linear_no_bias(
                    head_width,
                    SQUARES * SQUARES,
                    vb.pp("smolgen_projection"),
                )?),
                SmolgenConfig::Disabled {} => None,
            },
            policy_query: linear(config.width, config.policy_width, vb.pp("policy_query"))?,
            policy_key: linear(config.width, config.policy_width, vb.pp("policy_key"))?,
            promotion: linear_no_bias(config.policy_width, 4, vb.pp("promotion"))?,
            value: ValueHead::new(vb, config)?,
        })
    }

    pub(crate) fn cpu_piece_value(&self) -> Result<Option<CpuPieceValue>> {
        self.value.cpu_piece_value()
    }

    pub fn forward(
        &self,
        position: &Tensor,
        move_index: &Tensor,
        promotion_index: &Tensor,
        promotes: &Tensor,
        pieces: Option<&Tensor>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let encoded = self.encode(position)?;
        let policy = self.legal_policy(&encoded, move_index, promotion_index, promotes)?;
        let wdl = self.value.logits(&encoded, pieces)?;
        Ok((policy, wdl))
    }

    fn encode(&self, position: &Tensor) -> Result<Tensor> {
        let mut x = self.input.forward(position)?.broadcast_add(&self.square)?;
        for layer in &self.layers {
            x = layer.forward(&x, self.smolgen_projection.as_ref(), self.config)?;
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
