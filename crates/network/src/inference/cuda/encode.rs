use std::sync::OnceLock;

use crate::{INPUTS, SQUARES};
use candle_core::cuda_backend::cudarc::{
    driver::{LaunchConfig, PushKernelArg},
    nvrtc::{CompileOptions, compile_ptx_with_opts},
};
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
        static PTX: OnceLock<std::result::Result<String, String>> = OnceLock::new();
        let ptx = PTX
            .get_or_init(|| {
                compile_ptx_with_opts(
                    include_str!("encode.cu"),
                    CompileOptions {
                        fmad: Some(false),
                        ..Default::default()
                    },
                )
                .map(|ptx| ptx.to_src())
                .map_err(|e| e.to_string())
            })
            .as_ref()
            .map_err(|e| candle_core::Error::Msg(e.clone()))?;
        let device = &storage.device;
        let function = device.get_or_load_custom_func("encode_boards", "ce5_encode_boards", ptx)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    #[ignore = "requires CUDA"]
    fn packed_encoding_matches_cpu_exactly() -> Result<()> {
        let mut packed = Vec::new();
        let mut expected = Vec::new();
        for i in 0..256usize {
            let board = std::array::from_fn(|s| (((s + i) % 13) | (((s * 7 + i) % 13) << 4)) as u8);
            let flags = [
                (i % 2) as u8,
                (i % 16) as u8,
                (i % 65) as u8,
                i as u8,
                (i % 4) as u8,
            ];
            packed.extend(board);
            packed.extend(flags);
            crate::encode_position(
                &mut expected,
                &board,
                flags[0] != 0,
                flags[1],
                flags[2],
                flags[3],
                flags[4],
            )
            .unwrap();
        }
        let actual = Tensor::from_vec(packed, (256, PACKED), &Device::new_cuda(0)?)?
            .apply_op1_no_bwd(&Encode)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(actual, expected);
        Ok(())
    }
}
