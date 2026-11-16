# Material hint ablation

Test whether removing material hints preserves playing strength while simplifying the model.

Reuse the no-hint control trained in the material-hint sweep.

| Hints | Policy loss | WDL loss | Total loss |
|---|---:|---:|---:|
| Enabled | 1.939953 | 0.686767 | 2.626720 |
| Disabled | 1.938736 | 0.690366 | 2.629102 |

The no-hint candidate used 309.04s timed training runtime and 3286.51s generation charge, totaling 3595.55s weighted budget.

Non-regression SPRT [−20, 0] nElo **passed** after **272 game pairs**: W/D/L **179/215/150**, **+19.9 nElo (95% CI −10.5, +50.0)**. Inference measured 2.162ms candidate / 2.331ms baseline at batch 512, giving 862 / 800 time-matched visits per move.

Promoted the no-hint candidate and disabled hints in the training recipe.

Candidate: `/data/chess/chess-engine-5/models/material-hint-tuning-2026.11.15/cosine-002-no-hints`.

Match artifacts: `/data/chess/chess-engine-5/models/material-hint-ablation-2026.11.15`.
