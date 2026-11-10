use candle_core::{Device, Result, Tensor};

use crate::{INPUTS, Net, SQUARES, move_indices};

pub struct PolicyInputs {
    positions: Tensor,
    moves: Tensor,
    promotions: Tensor,
    promotes: Tensor,
}

impl PolicyInputs {
    pub fn new(
        positions: Vec<f32>,
        moves: impl ExactSizeIterator<Item = impl ExactSizeIterator<Item = u16>> + Clone,
        device: &Device,
    ) -> Result<Self> {
        let size = moves.len();
        let width = moves.clone().map(|row| row.len()).max().unwrap_or(0);
        let mut indices = vec![0; size * width];
        let mut promotions = indices.clone();
        let mut promotes = vec![0.; size * width];
        for (i, row) in moves.enumerate() {
            for (j, mv) in row.enumerate() {
                let k = i * width + j;
                (indices[k], promotions[k], promotes[k]) = move_indices(mv);
            }
        }
        let shape = (size, width);
        Ok(Self {
            positions: Tensor::from_vec(positions, (size, SQUARES, INPUTS), device)?,
            moves: Tensor::from_vec(indices, shape, device)?,
            promotions: Tensor::from_vec(promotions, shape, device)?,
            promotes: Tensor::from_vec(promotes, shape, device)?,
        })
    }

    pub fn move_width(&self) -> usize {
        self.moves.dims()[1]
    }

    pub fn logits(&self, net: &Net, pieces: Option<&Tensor>) -> Result<(Tensor, Option<Tensor>)> {
        net.forward(
            &self.positions,
            &self.moves,
            &self.promotions,
            &self.promotes,
            pieces,
        )
    }
}
