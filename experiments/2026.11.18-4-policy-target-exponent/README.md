# Policy target sharpness

Raise training visit counts to a power before normalization. Evaluate on the original search distribution.

Baseline held-out total loss: 2.629102.

| Setting | Policy | WDL | Total loss | Runtime (s) | Generation (s) | Total budget (s) |
|---|---:|---:|---:|---:|---:|---:|
| exponent-0.8 | 1.944520 | 0.691411 | 2.635931 | 304.27 | 3286.51 | 3590.79 |
| exponent-1.25 | 1.957481 | 0.693257 | 2.650738 | 306.09 | 3286.51 | 3592.60 |

No candidate improved held-out total loss by the 0.002496 match threshold.

Artifacts: `/data/chess/chess-engine-5/models/supervision-queue-2026.11.18/policy-target-exponent`.
