# Batch size and learning rate

Tune batch size jointly with learning rate within the 60-minute budget.

Keep training positions fixed. Keep epochs fixed where the budget allows; calibrate epochs using two-epoch timing probes otherwise. Timing probes are excluded from the candidate table.

| Batch | Peak LR | Epochs | Policy loss | WDL loss | Total loss |
|---|---|---|---|---|---|
| 128 | 0.002 | 15.0 | 1.938736 | 0.690366 | 2.629102 |
| 64 | 0.001 | 7.4875 | 1.955184 | 0.696970 | 2.652154 |
| 64 | 0.002 | 7.4875 | 1.942487 | 0.697794 | 2.640281 |
| 64 | 0.004 | 7.4875 | 1.951597 | 0.700655 | 2.652252 |
| 128 | 0.001 | 15.0 | 1.951925 | 0.690239 | 2.642164 |
| 128 | 0.004 | 15.0 | 1.939303 | 0.692850 | 2.632153 |
| 256 | 0.001 | 15.0 | 1.969959 | 0.691569 | 2.661529 |
| 256 | 0.002 | 15.0 | 1.941227 | 0.690411 | 2.631638 |
| 256 | 0.004 | 15.0 | 1.939711 | 0.692396 | 2.632107 |

The baseline had the lowest total loss. Selected `batch-256-lr-0.002` as the best new candidate for the requested match.

Timed runtime: 173.19–311.72s. Generation charge: 3286.51s per run. Weighted budget: 3459.70–3598.23s. Selected candidate: 3459.70s.

Matched-speed gainer SPRT [0, 20] nElo **failed** after **877 game pairs**, at 800 visits per move for both models. W/D/L **552/649/553**, **+3.0 (-14.2, +21.2) nElo (95% CI)**.

Kept the baseline recipe.

Artifacts: `/data/chess/chess-engine-5/models/training-sweeps-2026.11.15/batch-learning-rate`.
