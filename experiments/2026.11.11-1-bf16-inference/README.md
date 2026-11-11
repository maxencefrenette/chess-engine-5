# BF16 inference

Test whether BF16 transformer inference gains Elo by spending its speedup on more search visits.

Reuse the promoted baseline's weights exactly. Transformer inference uses BF16; the small CPU value MLP stays FP32. No retraining was needed. The shared checkpoint's training cost was 287.60s timed runtime plus 0.3 × 5,020.93s estimated generation, totaling 1,793.88 weighted seconds.

| Inference batch | FP32 (ms) | BF16 (ms) | Throughput gain |
|---:|---:|---:|---:|
| 256 | 1.500 | 1.338 | 12.1% |
| 512 | 3.041 | 2.525 | 20.4% |
| 1024 | 6.412 | 5.141 | 24.7% |

Two runs per precision, in FP32/BF16/BF16/FP32 order, with 20 warmups and 500 timed evaluations per batch size. These timings include input preparation, transfers, CPU value evaluation, and policy normalization. FP32 predictions remain exactly unchanged by the precision support code.

- W/D/L 92/103/53 over 248 games
- Observed Elo +55.1
- SPRT verdict Passed ([0, 20], final LLR +3.0124)
- Visits per move: 951 BF16 candidate, 800 FP32 baseline

Match calibration at batch 512 measured 2.694 ms for BF16 and 3.201 ms for FP32. The BF16 checkpoint is now the baseline, and `model.inference_precision = "bf16"` is carried forward in the training recipe. Training arithmetic remains FP32.

Artifacts and match log: `/data/chess/chess-engine-5/models/speedups-2026.11.10/bf16-benchmark`.
