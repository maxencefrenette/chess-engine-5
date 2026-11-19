//! Training-only policy prediction for the position after the recorded move.
use candle_core::{Device, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear, ops::log_softmax};
use std::collections::HashMap;

pub struct Reply {
    played: u16,
    moves: Vec<(u16, u16)>,
}

pub fn targets(selected: &[chess_data::Row]) -> Vec<Option<Reply>> {
    // Use only selected rows, so every policy we train on incurred its generation charge.
    let lookup: HashMap<_, _> = selected
        .iter()
        .map(|r| ((r.game_id, r.position.ply), &r.position.policy))
        .collect();
    selected
        .iter()
        .map(|r| {
            let next = r.position.ply.checked_add(1)?;
            Some(Reply {
                played: r.position.played_move,
                moves: (*lookup.get(&(r.game_id, next))?).clone(),
            })
        })
        .collect()
}

pub struct Targets {
    from: Tensor,
    to: Tensor,
    promo: Tensor,
    moves: Tensor,
    offsets: Tensor,
    promotes: Tensor,
    distribution: Tensor,
    mask: Tensor,
    count: usize,
}

impl Targets {
    pub fn batch<'a>(
        rows: impl ExactSizeIterator<Item = (Option<&'a Reply>, u16)> + Clone,
        device: &Device,
    ) -> Result<Option<Self>> {
        let size = rows.len();
        let width = rows
            .clone()
            .filter_map(|(r, _)| r.map(|r| r.moves.len()))
            .max()
            .unwrap_or(0);
        if width == 0 {
            return Ok(None);
        }
        let mut from = vec![0u32; size];
        let mut to = from.clone();
        let mut promo = vec![0f32; size * 5];
        let mut moves = vec![0u32; size * width];
        let mut offsets = moves.clone();
        let mut promotes = vec![0f32; size * width];
        let mut distribution = promotes.clone();
        let mut mask = vec![-1e9f32; size * width];
        let mut count = 0;
        for (i, (reply, flip)) in rows.enumerate() {
            let Some(reply) = reply else { continue };
            count += 1;
            let played = reply.played ^ flip;
            from[i] = (i * 64 + usize::from(played & 63)) as u32;
            to[i] = (i * 64 + usize::from((played >> 6) & 63)) as u32;
            promo[i * 5 + usize::from(played >> 12)] = 1.;
            let total: f32 = reply.moves.iter().map(|m| f32::from(m.1)).sum();
            for (j, &(mv, visits)) in reply.moves.iter().enumerate() {
                let k = i * width + j;
                (moves[k], offsets[k], promotes[k]) = chess_network::move_indices(mv ^ flip);
                mask[k] = 0.;
                distribution[k] = f32::from(visits) / total;
            }
        }
        let shape = (size, width);
        Ok(Some(Self {
            from: Tensor::from_vec(from, size, device)?,
            to: Tensor::from_vec(to, size, device)?,
            promo: Tensor::from_vec(promo, (size, 5), device)?,
            moves: Tensor::from_vec(moves, shape, device)?,
            offsets: Tensor::from_vec(offsets, shape, device)?,
            promotes: Tensor::from_vec(promotes, shape, device)?,
            distribution: Tensor::from_vec(distribution, shape, device)?,
            mask: Tensor::from_vec(mask, shape, device)?,
            count,
        }))
    }
}

pub struct Head {
    hidden: Linear,
    logits: Linear,
}

impl Head {
    pub fn new(vb: VarBuilder, width: usize) -> Result<Self> {
        Ok(Self {
            hidden: linear(3 * width + 5, width, vb.pp("hidden"))?,
            logits: linear(width, 4096 + 256, vb.pp("policy"))?,
        })
    }
    pub fn loss(&self, encoded: &Tensor, target: &Targets) -> Result<Tensor> {
        let flat = encoded.flatten(0, 1)?;
        let context = Tensor::cat(
            &[
                &encoded.mean(1)?,
                &flat.index_select(&target.from, 0)?,
                &flat.index_select(&target.to, 0)?,
                &target.promo,
            ],
            1,
        )?;
        let logits = self
            .logits
            .forward(&self.hidden.forward(&context)?.relu()?)?;
        let main = logits
            .narrow(1, 0, 4096)?
            .contiguous()?
            .gather(&target.moves, 1)?;
        let offset = logits
            .narrow(1, 4096, 256)?
            .contiguous()?
            .gather(&target.offsets, 1)?;
        let legal = (main + (offset * &target.promotes)?)?;
        let logp = log_softmax(&(&legal + &target.mask)?, 1)?;
        (logp * &target.distribution)?
            .sum_all()?
            .affine(-1. / target.count as f64, 0.)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Var};
    use candle_nn::VarMap;
    #[test]
    fn replies_train_the_parent_features_and_ignore_missing_rows() -> Result<()> {
        let device = &Device::Cpu;
        let head = Head::new(
            VarBuilder::from_varmap(&VarMap::new(), DType::F32, device),
            8,
        )?;
        let reply = Reply {
            played: 12 | (28 << 6),
            moves: vec![(52 | (36 << 6), 9), (51 | (35 << 6), 1)],
        };
        let batch = Targets::batch([(Some(&reply), 0), (None, 0)].into_iter(), device)?.unwrap();
        let features = Var::randn(0f32, 1., (2, 64, 8), device)?;
        let grads = head.loss(&features, &batch)?.backward()?;
        let grad = grads.get(&features).unwrap();
        assert!(grad.get(0)?.abs()?.sum_all()?.to_scalar::<f32>()? > 0.);
        assert_eq!(grad.get(1)?.abs()?.sum_all()?.to_scalar::<f32>()?, 0.);
        Ok(())
    }
}
