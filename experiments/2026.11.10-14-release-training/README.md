# Release training

Reduce CPU overhead by building the trainer in release mode.

Fixed-workload benchmark: 250,301 positions, one epoch (1,956 steps), seed 42; debug/release/release/debug order. These are implementation benchmarks, not budget-matched candidate runs.

| Build | Training time, two runs (s) | Held-out evaluation (s) |
|---|---|---|
| Debug | 30.70, 30.48 | 0.497, 0.498 |
| Release | 16.14, 16.31 | 0.274, 0.274 |

Training time fell 47.0%. Compared checkpoint tensors were exactly equal; all runs had identical held-out policy and WDL losses. Accepted: `mise run train` now uses release mode. No baseline match was required.

Artifacts: `/data/chess/chess-engine-5/models/speedups-2026.11.10`.
