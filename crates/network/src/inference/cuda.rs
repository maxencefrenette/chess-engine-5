use std::collections::{HashMap, VecDeque};

use candle_core::cuda_backend::{
    CudaDType,
    cudarc::{
        driver::result::DriverError,
        driver::{CudaGraph, DeviceRepr, sys},
        nvrtc::{CompileOptions, compile_ptx_with_opts},
    },
};
use candle_core::{CpuStorage, CudaStorage, DType, Device, InplaceOp1, Layout, Result, Tensor};

use crate::Net;

mod encode;
mod policy;
use encode::{DecodeMoves, Encode, PACKED};

type PtxCache = std::sync::OnceLock<std::result::Result<String, String>>;

fn compile(
    cache: &'static PtxCache,
    source: &str,
    arch: Option<&'static str>,
) -> Result<&'static str> {
    cache
        .get_or_init(|| {
            compile_ptx_with_opts(
                source,
                CompileOptions {
                    fmad: Some(false),
                    arch,
                    ..Default::default()
                },
            )
            .map(|ptx| ptx.to_src())
            .map_err(|e| e.to_string())
        })
        .as_deref()
        .map_err(|e| candle_core::Error::Msg(e.clone()))
}

type Logits = (Vec<Vec<f32>>, Vec<Vec<f32>>);

// Retain only a few shapes: game completion changes the active batch size.
#[derive(Default)]
pub(super) struct Inference {
    entries: VecDeque<Entry>,
    shapes: GraphShapes,
}

const HOT_SHAPES: usize = 8;
const SHAPE_WINDOW: usize = 2048;

// Re-estimate which irregular shapes justify capturing a graph as games progress.
#[derive(Default)]
struct GraphShapes {
    counts: HashMap<(usize, usize), usize>,
    calls: usize,
    frequent: [(usize, usize); HOT_SHAPES],
}

impl GraphShapes {
    fn observe(&mut self, shape: (usize, usize)) -> bool {
        let regular = shape.0.is_power_of_two();
        if !regular {
            *self.counts.entry(shape).or_default() += 1;
        }
        self.calls += 1;
        if self.calls == SHAPE_WINDOW {
            let mut counts: Vec<_> = self.counts.drain().collect();
            counts.sort_unstable_by_key(|&(shape, count)| (std::cmp::Reverse(count), shape));
            self.frequent = std::array::from_fn(|i| {
                counts
                    .get(i)
                    .filter(|(_, count)| *count >= 32)
                    .map_or((0, 0), |&(shape, _)| shape)
            });
            self.calls = 0;
        }
        regular || self.frequent.contains(&shape)
    }
}

struct Entry {
    graph: CudaGraph,
    inputs: [Tensor; 2],
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
        let mut packed_moves = vec![0u8; batch * width * 2];
        for (i, ((board, black), moves)) in positions.zip(moves).enumerate() {
            let (board, black, mask) = net.config.perspective.orient(board, black);
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
            for (j, mv) in moves.enumerate() {
                let offset = 2 * (i * width + j);
                packed_moves[offset..offset + 2].copy_from_slice(&(mv ^ mask).to_le_bytes());
            }
        }
        // Use exact sizes because batch padding changes BF16 GEMM results.
        let shape = (batch, width);
        if !self.shapes.observe(shape) {
            let inputs = [
                Tensor::from_vec(boards, (batch, PACKED), device)?,
                Tensor::from_vec(packed_moves, (batch, width, 2), device)?,
            ];
            let (policy, value) = forward(net, &inputs)?;
            return Ok((policy.to_vec2()?, value.to_vec2()?));
        }
        let found = self.entries.iter().position(|entry| entry.shape == shape);
        let entry = match found {
            Some(index) => self.entries.remove(index).unwrap(),
            None => {
                if self.entries.len() == 4 + HOT_SHAPES {
                    self.entries.pop_front();
                }
                Entry::new(net, device, shape)?
            }
        };
        self.entries.push_back(entry);
        let entry = self.entries.back().unwrap();
        entry.inputs[0].inplace_op1(&Upload(&boards))?;
        entry.inputs[1].inplace_op1(&Upload(&packed_moves))?;
        entry.graph.launch().map_err(cuda_error)?;
        Ok((entry.outputs.0.to_vec2()?, entry.outputs.1.to_vec2()?))
    }
}

impl Entry {
    fn new(net: &Net, device: &Device, shape: (usize, usize)) -> Result<Self> {
        let (batch, width) = shape;
        let inputs = [
            Tensor::zeros((batch, PACKED), DType::U8, device)?,
            Tensor::zeros((batch, width, 2), DType::U8, device)?,
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

fn forward(net: &Net, inputs: &[Tensor; 2]) -> Result<(Tensor, Tensor)> {
    let positions = inputs[0].apply_op1_no_bwd(&Encode)?;
    let moves = inputs[1].apply_op1_no_bwd(&DecodeMoves)?;
    let (indices, promotions, promotes) = (moves.get(0)?, moves.get(1)?, moves.get(2)?);
    let (policy, value) =
        if net.config.inference_precision.dtype() == DType::BF16 && net.config.policy_width == 32 {
            let encoded = net.encode(&positions)?;
            let policy = net.policy_from_scores(&encoded, &promotions, &promotes, |q, k| {
                q.apply_op3_no_bwd(k, &indices, &policy::LegalDots)?
                    .affine(32f64.sqrt().recip(), 0.)
            })?;
            (policy, net.value.logits(&encoded)?)
        } else {
            net.forward(&positions, &indices, &promotions, &promotes)?
        };
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

#[cfg(test)]
mod tests {
    use super::{GraphShapes, SHAPE_WINDOW};

    #[test]
    fn graph_admission_tracks_frequent_exact_shapes_as_the_workload_changes() {
        let mut shapes = GraphShapes::default();
        assert!(shapes.observe((256, 64)));
        assert!(!shapes.observe((255, 64)));
        for _ in 2..SHAPE_WINDOW {
            shapes.observe((253, 64));
        }
        assert!(shapes.observe((253, 64)));
        assert!(!shapes.observe((253, 32)));
        for _ in 2..SHAPE_WINDOW {
            shapes.observe((249, 32));
        }
        assert!(shapes.observe((249, 32)));
        assert!(!shapes.observe((253, 64)));
    }
}
