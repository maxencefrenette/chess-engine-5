use candle_core::cuda_backend::{
    CudaStorageSlice,
    cudarc::driver::{LaunchConfig, PushKernelArg},
};
use candle_core::{CpuStorage, CudaStorage, CustomOp3, Layout, Result, Shape};

/// The indices come from DecodeMoves, which bounds both squares to 0..64.
pub(super) struct LegalDots;

impl CustomOp3 for LegalDots {
    fn name(&self) -> &'static str {
        "legal-policy-dots"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("legal-policy kernel requires CUDA")
    }

    fn cuda_fwd(
        &self,
        q: &CudaStorage,
        q_layout: &Layout,
        k: &CudaStorage,
        k_layout: &Layout,
        ids: &CudaStorage,
        id_layout: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let (batch, width) = id_layout.shape().dims2()?;
        if q_layout.shape().dims() != [batch, 64, 32]
            || q_layout.shape() != k_layout.shape()
            || [q_layout, k_layout, id_layout]
                .iter()
                .any(|l| !l.is_contiguous() || l.start_offset() != 0)
        {
            candle_core::bail!("invalid packed policy shape")
        }
        let (CudaStorageSlice::BF16(queries), CudaStorageSlice::BF16(keys)) = (&q.slice, &k.slice)
        else {
            candle_core::bail!("legal-policy kernel requires BF16")
        };
        let count = u32::try_from(id_layout.shape().elem_count())
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let threads = count
            .div_ceil(8)
            .checked_mul(32)
            .ok_or_else(|| candle_core::Error::Msg("policy batch too large".into()))?;
        let width = width as u32;
        let device = &q.device;
        static PTX: super::PtxCache = super::PtxCache::new();
        let kernel = device.get_or_load_custom_func(
            "legal_policy_dots",
            "ce5_policy",
            super::compile(&PTX, include_str!("policy.cu"), Some("compute_80"))?,
        )?;
        let output = unsafe { device.alloc(count as usize)? };
        let mut launch = kernel.builder();
        launch
            .arg(queries)
            .arg(keys)
            .arg(ids.as_cuda_slice::<u32>()?)
            .arg(&output)
            .arg(&count)
            .arg(&width);
        // Each valid index has one writer. The decoder supplies a 12-bit square pair.
        unsafe { launch.launch(LaunchConfig::for_num_elems(threads)) }
            .map_err(super::cuda_error)?;
        Ok((
            CudaStorage {
                slice: CudaStorageSlice::BF16(output),
                device: device.clone(),
            },
            id_layout.shape().clone(),
        ))
    }
}
