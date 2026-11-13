use std::collections::VecDeque;

use candle_core::cuda_backend::{
    CudaDType,
    cudarc::{
        driver::result::DriverError,
        driver::{CudaGraph, DeviceRepr, sys},
    },
};
use candle_core::{CpuStorage, CudaStorage, DType, Device, InplaceOp1, Layout, Result, Tensor};

use crate::{Net, move_indices};

mod encode;
use encode::{Encode, PACKED};

type Logits = (Vec<Vec<f32>>, Vec<Vec<f32>>);

// Retain only a few shapes: game completion changes the active batch size.
#[derive(Default)]
pub(super) struct Inference {
    entries: VecDeque<Entry>,
}

struct Entry {
    graph: CudaGraph,
    inputs: [Tensor; 4],
    outputs: (Tensor, Tensor),
    shape: (usize, usize),
}

impl Inference {
    pub(super) fn evaluate(
        &mut self,
        net: &Net,
        positions: impl ExactSizeIterator<Item = (chess_data::Board, bool)>,
        moves: impl ExactSizeIterator<Item = impl ExactSizeIterator<Item = u16>> + Clone,
        device: &Device,
    ) -> Result<Logits> {
        let batch = moves.len();
        let width = moves
            .clone()
            .map(|m| m.len())
            .max()
            .unwrap_or(1)
            .max(1)
            .next_power_of_two();
        let mut boards = vec![0u8; batch * PACKED];
        for (i, (board, black)) in positions.enumerate() {
            if board.board.iter().any(|&b| (b & 15) > 12 || (b >> 4) > 12) {
                candle_core::bail!("invalid piece code")
            }
            let row = &mut boards[i * PACKED..(i + 1) * PACKED];
            row[..32].copy_from_slice(&board.board);
            row[32..].copy_from_slice(&[
                u8::from(black),
                board.castling,
                board.ep_square,
                board.halfmove_clock,
                board.repetitions,
            ]);
        }
        let mut indices = vec![0u32; batch * width];
        let mut promotions = indices.clone();
        let mut promotes = vec![0f32; batch * width];
        for (i, row) in moves.enumerate() {
            for (j, mv) in row.enumerate() {
                (
                    indices[i * width + j],
                    promotions[i * width + j],
                    promotes[i * width + j],
                ) = move_indices(mv);
            }
        }
        // Padding the batch dimension changes BF16 GEMM results. Capture common
        // exact sizes; irregular batches use eager inference without padding.
        if !batch.is_power_of_two() {
            let inputs = [
                Tensor::from_vec(boards, (batch, PACKED), device)?,
                Tensor::from_vec(indices, (batch, width), device)?,
                Tensor::from_vec(promotions, (batch, width), device)?,
                Tensor::from_vec(promotes, (batch, width), device)?,
            ];
            let (policy, value) = forward(net, &inputs)?;
            return Ok((policy.to_vec2()?, value.to_vec2()?));
        }
        let shape = (batch, width);
        let found = self.entries.iter().position(|entry| entry.shape == shape);
        let entry = match found {
            Some(index) => self.entries.remove(index).unwrap(),
            None => {
                if self.entries.len() == 4 {
                    self.entries.pop_front();
                }
                Entry::new(net, device, shape)?
            }
        };
        self.entries.push_back(entry);
        let entry = self.entries.back().unwrap();
        entry.inputs[0].inplace_op1(&Upload(&boards))?;
        entry.inputs[1].inplace_op1(&Upload(&indices))?;
        entry.inputs[2].inplace_op1(&Upload(&promotions))?;
        entry.inputs[3].inplace_op1(&Upload(&promotes))?;
        entry.graph.launch().map_err(cuda_error)?;
        Ok((entry.outputs.0.to_vec2()?, entry.outputs.1.to_vec2()?))
    }
}

impl Entry {
    fn new(net: &Net, device: &Device, shape: (usize, usize)) -> Result<Self> {
        let (batch, width) = shape;
        let inputs = [
            Tensor::zeros((batch, PACKED), DType::U8, device)?,
            Tensor::zeros((batch, width), DType::U32, device)?,
            Tensor::zeros((batch, width), DType::U32, device)?,
            Tensor::zeros((batch, width), DType::F32, device)?,
        ];
        let cuda = device.as_cuda_device()?;
        let _cache = cuda.enable_cuda_graph_htod_cache();
        // Warm kernels and cache Candle's small shape-metadata uploads before capture.
        let _ = forward(net, &inputs)?;
        device.synchronize()?;
        let stream = cuda.cuda_stream();
        stream
            .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .map_err(cuda_error)?;
        let outputs = forward(net, &inputs);
        // Always end capture, including when a forward operation fails.
        let graph = stream
            .end_capture(
                sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            )
            .map_err(cuda_error);
        let outputs = outputs?;
        let graph =
            graph?.ok_or_else(|| candle_core::Error::Msg("empty inference graph".into()))?;
        Ok(Self {
            graph,
            inputs,
            outputs,
            shape,
        })
    }
}

fn forward(net: &Net, inputs: &[Tensor; 4]) -> Result<(Tensor, Tensor)> {
    let positions = inputs[0].apply_op1_no_bwd(&Encode)?;
    let (policy, value) = net.forward(&positions, &inputs[1], &inputs[2], &inputs[3])?;
    Ok((policy.to_dtype(DType::F32)?, value.to_dtype(DType::F32)?))
}

fn cuda_error(error: DriverError) -> candle_core::Error {
    candle_core::Error::Msg(error.to_string())
}

struct Upload<'a, T>(&'a [T]);

impl<T: CudaDType + DeviceRepr + 'static> InplaceOp1 for Upload<'_, T> {
    fn name(&self) -> &'static str {
        "inference-input-upload"
    }
    fn cpu_fwd(&self, _: &mut CpuStorage, _: &Layout) -> Result<()> {
        candle_core::bail!("CUDA upload requires CUDA storage")
    }
    fn cuda_fwd(&self, storage: &mut CudaStorage, layout: &Layout) -> Result<()> {
        if !layout.is_contiguous()
            || layout.start_offset() != 0
            || layout.shape().elem_count() != self.0.len()
        {
            candle_core::bail!("inference upload shape mismatch")
        }
        let device = storage.device.clone();
        device.memcpy_htod(self.0, storage.as_cuda_slice_mut::<T>()?)
    }
}
