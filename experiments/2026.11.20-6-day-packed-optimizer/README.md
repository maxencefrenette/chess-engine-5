# Packed FP32 AdamW

Measure the training speedup and test a 60-minute candidate trained on more positions.

| Candidate | Positions | Policy loss | Mixed WDL loss | Total loss | Training (s) | Generation (s) | Budget (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP32 control | 370,100 | 1.68436 | 0.69684 | 2.38120 | 440.67 | 3155.95 | 3596.63 |
| Packed FP32 | 383,300 | 1.68476 | 0.69077 | 2.37553 | 331.08 | 3268.51 | 3599.58 |

Packing AdamW updates reduced FP32 step time from 10.37 to 7.37 ms. After 1,012 identical steps all 35 saved tensors matched bit-for-bit. The savings funded 13,200 more unique positions at 15 epochs.

| Candidate | Opponent | Visits (candidate/baseline) | W/D/L | nElo (95% CI) | Non-regression SPRT |
|---|---|---:|---|---|---|
| Packed FP32 | FP32 control | 800/800 | 473/374/461 | +0.6 (-20.0, +20.3) | passed |

Promoted the 383,300-position checkpoint.

## BF16 investigation

| Implementation | FP32 (ms) | Mixed BF16 (ms) |
|---|---:|---:|
| Per-tensor AdamW | 10.31–10.37 | 11.52–11.64 |
| Packed AdamW, resident compute weights | 7.37–7.43 | 7.68–7.77 |

Each timing covers 1,012 optimizer steps at batch 128 on the RTX 5070. In each precision, all 35 checkpoint tensors from packed AdamW matched per-tensor AdamW bit-for-bit. Asking cuBLAS to use fast BF16 math only for matrix multiplies with FP32 storage took 7.33–7.46 ms with packed AdamW.

Over 160 traced steps, summed matrix-kernel time fell from 0.56 to 0.28 ms with BF16. With per-tensor AdamW, BF16 dispatched 1,485 GPU kernels per step versus 1,341 in FP32, including 113 versus 8 cast kernels. The largest observed speedup came from packing the optimizer updates.

The mixed-BF16 candidate trained on 363,800 positions (training 483.91s, generation 3102.24s, total 3586.15s). Its gainer was stopped by request. At the last reported 290 pairs, W/D/L was 229/173/178 and nElo was +35.7 (+6.8, +64.2), 95% CI. No SPRT verdict.
