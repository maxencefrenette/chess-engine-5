# Weight averaging

Test whether averaging late-training weights improves held-out loss and playing strength.

Average post-update weights every 10 optimizer steps over the final 10% or 25% of training, including the final step. Averaging leaves live optimizer weights untouched; held-out evaluation and checkpoint saving use the averaged weights.

| Tail fraction | Policy loss | WDL loss | Total loss |
|---|---|---|---|
| 0 | 1.938736 | 0.690366 | 2.629102 |
| 0.1 | 1.935773 | 0.689466 | 2.625240 |
| 0.25 | 1.935766 | 0.689648 | 2.625413 |

Selected `tail-0.1` by total held-out loss.

Timed runtime: 311.39–313.48s. Generation charge: 3286.51s per run. Weighted budget: 3597.90–3599.99s. Selected candidate: 3597.90s.

Matched-speed gainer SPRT [0, 20] nElo **failed** after **971 game pairs**, at 800 visits per move for both models. W/D/L **594/757/591**, **+4.0 (-12.4, +21.5) nElo (95% CI)**.

Kept the baseline recipe.

Artifacts: `/data/chess/chess-engine-5/models/training-sweeps-2026.11.15/weight-averaging`.
