# Weight decay

Tune AdamW weight decay within the 60-minute budget.

Sweep weight decay 0, 0.01, 0.05, and 0.1, reusing the current setting as the control.

| Weight decay | Policy loss | WDL loss | Total loss |
|---|---|---|---|
| 0.01 | 1.938736 | 0.690366 | 2.629102 |
| 0.0 | 1.939876 | 0.690978 | 2.630854 |
| 0.05 | 1.939249 | 0.692075 | 2.631324 |
| 0.1 | 1.947845 | 0.697593 | 2.645438 |

The baseline had the lowest total loss. Selected `decay-0.0` as the best new candidate for the requested match.

Timed runtime: 309.46–314.16s. Generation charge: 3286.51s per run. Weighted budget: 3595.97–3600.67s. Selected candidate: 3595.97s.

Matched-speed gainer SPRT [0, 20] nElo **failed** after **363 game pairs**, at 800 visits per move for both models. W/D/L **213/278/235**, **-11.6 (-37.7, +15.0) nElo (95% CI)**.

Kept the baseline recipe.

Artifacts: `/data/chess/chess-engine-5/models/training-sweeps-2026.11.15/weight-decay`.
