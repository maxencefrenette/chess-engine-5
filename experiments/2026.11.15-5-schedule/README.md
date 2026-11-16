# Learning-rate schedule

Tune peak learning rate and the final cosine learning-rate ratio within the 60-minute budget.

Cosine schedules use 5% warmup. The exact 0.002 / 0.1 control reuses the existing checkpoint.

| Peak LR | Final ratio | Policy loss | WDL loss | Total loss |
|---|---|---|---|---|
| 0.001 | 0.0 | 1.954285 | 0.691217 | 2.645502 |
| 0.001 | 0.1 | 1.951925 | 0.690239 | 2.642164 |
| 0.001 | 0.3 | 1.951075 | 0.694012 | 2.645088 |
| 0.002 | 0.0 | 1.936557 | 0.689668 | 2.626225 |
| 0.002 | 0.1 | 1.938736 | 0.690366 | 2.629102 |
| 0.002 | 0.3 | 1.945559 | 0.700077 | 2.645636 |
| 0.004 | 0.0 | 1.935757 | 0.690513 | 2.626270 |
| 0.004 | 0.1 | 1.939303 | 0.692850 | 2.632153 |
| 0.004 | 0.3 | 1.950040 | 0.706051 | 2.656090 |

Selected `lr-0.002-floor-0.0` by total held-out loss.

Timed runtime: 308.31–312.80s. Generation charge: 3286.51s per run. Weighted budget: 3594.82–3599.31s. Selected candidate: 3597.25s.

Matched-speed gainer SPRT [0, 20] nElo **failed** after **1,655 game pairs**, at 800 visits per move for both models. W/D/L **1037/1269/1004**, **+7.6 (-5.7, +22.3) nElo (95% CI)**.

Kept the baseline recipe.

Artifacts: `/data/chess/chess-engine-5/models/training-sweeps-2026.11.15/schedule`.
