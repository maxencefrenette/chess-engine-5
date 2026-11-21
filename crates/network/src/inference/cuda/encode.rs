use crate::{INPUTS, SQUARES};
use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
use candle_core::{CpuStorage, CudaStorage, CustomOp1, Layout, Result, Shape};

pub(super) const PACKED: usize = 37;
// Keep the CUDA kernel's indexing in sync with the board encoding.
const _: () = assert!(INPUTS == 21 && SQUARES == 64 && PACKED == 37);
pub(super) struct Encode;

impl CustomOp1 for Encode {
    fn name(&self) -> &'static str {
        "encode-packed-boards"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("packed-board encoder requires CUDA")
    }
    fn cuda_fwd(&self, storage: &CudaStorage, layout: &Layout) -> Result<(CudaStorage, Shape)> {
        let (batch, width) = layout.shape().dims2()?;
        if width != PACKED || !layout.is_contiguous() || layout.start_offset() != 0 {
            candle_core::bail!("invalid packed-board shape")
        }
        let device = &storage.device;
        let function = device.get_or_load_custom_func("encode_boards", "ce5_encode", ptx()?)?;
        let count = batch * SQUARES * INPUTS;
        let count_u32 = u32::try_from(count).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        // Every output element is written by the kernel below.
        let output = unsafe { device.alloc::<f32>(count)? };
        let mut launch = function.builder();
        launch
            .arg(storage.as_cuda_slice::<u8>()?)
            .arg(&output)
            .arg(&count_u32);
        // Source has batch * PACKED bytes; the kernel bounds-checks its output index.
        unsafe { launch.launch(LaunchConfig::for_num_elems(count_u32)) }
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        Ok((
            CudaStorage {
                slice: candle_core::cuda_backend::CudaStorageSlice::F32(output),
                device: device.clone(),
            },
            Shape::from((batch, SQUARES, INPUTS)),
        ))
    }
}

fn ptx() -> Result<&'static str> {
    static PTX: super::PtxCache = super::PtxCache::new();
    super::compile(&PTX, include_str!("encode.cu"), None)
}

pub(super) struct DecodeMoves;

impl CustomOp1 for DecodeMoves {
    fn name(&self) -> &'static str {
        "decode-packed-moves"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("packed move decoder requires CUDA")
    }
    fn cuda_fwd(&self, storage: &CudaStorage, layout: &Layout) -> Result<(CudaStorage, Shape)> {
        let (batch, width, bytes) = layout.shape().dims3()?;
        if bytes != 2 || !layout.is_contiguous() || layout.start_offset() != 0 {
            candle_core::bail!("invalid packed move shape")
        }
        let device = &storage.device;
        let count = u32::try_from(layout.shape().elem_count() / 2)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let function = device.get_or_load_custom_func("decode_moves", "ce5_encode", ptx()?)?;
        let output = unsafe { device.alloc::<u32>(3 * count as usize)? };
        let mut launch = function.builder();
        launch
            .arg(storage.as_cuda_slice::<u8>()?)
            .arg(&output)
            .arg(&count);
        // Each input ID produces one initialized element in each of the three output planes.
        unsafe { launch.launch(LaunchConfig::for_num_elems(count)) }
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        Ok((
            CudaStorage {
                slice: candle_core::cuda_backend::CudaStorageSlice::U32(output),
                device: device.clone(),
            },
            Shape::from((3, batch, width)),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    #[ignore = "requires CUDA"]
    fn packed_moves_match_cpu_indices_and_promotions() -> Result<()> {
        let moves: Vec<u16> = (0..5 * 64 * 64).collect();
        let packed = moves
            .iter()
            .flat_map(|mv| mv.to_le_bytes())
            .collect::<Vec<_>>();
        let actual = Tensor::from_vec(packed, (5, 64 * 64, 2), &Device::new_cuda(0)?)?
            .apply_op1_no_bwd(&DecodeMoves)?
            .reshape((3, moves.len()))?
            .to_vec2::<u32>()?;
        for (i, mv) in moves.into_iter().enumerate() {
            let (index, promotion, promotes) = crate::move_indices(mv);
            assert_eq!(
                [actual[0][i], actual[1][i], actual[2][i]],
                [index, promotion, promotes as u32]
            );
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires CUDA"]
    fn packed_encoding_matches_cpu_exactly() -> Result<()> {
        let mut packed = Vec::new();
        let mut expected = Vec::new();
        for i in 0..256usize {
            let snapshot = chess_data::Board {
                board: std::array::from_fn(|s| (((s + i) % 13) | (((s * 7 + i) % 13) << 4)) as u8),
                castling: (i % 16) as u8,
                ep_square: (i % 65) as u8,
                halfmove_clock: i as u8,
                repetitions: (i % 4) as u8,
            };
            let black = i % 2 == 1;
            packed.extend(snapshot.board);
            packed.extend([
                u8::from(black),
                snapshot.castling,
                snapshot.ep_square,
                snapshot.halfmove_clock,
                snapshot.repetitions,
            ]);
            crate::encode_position(&mut expected, &snapshot, black).unwrap();
        }
        let actual = Tensor::from_vec(packed, (256, PACKED), &Device::new_cuda(0)?)?
            .apply_op1_no_bwd(&Encode)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(actual, expected);
        Ok(())
    }
}
