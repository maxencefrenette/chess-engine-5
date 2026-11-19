# Spread WDL targets across each game

Select one WDL target at random from each equal-size segment of the selected positions in ply order. Keep the number of value targets per game fixed.

Baseline held-out total loss: 2.629102.

| Setting | Policy | WDL | Total loss | Runtime (s) | Generation (s) | Total budget (s) |
|---|---:|---:|---:|---:|---:|---:|
| stratified | 1.938981 | 0.689340 | 2.628321 | 307.33 | 3286.51 | 3593.84 |

No candidate improved held-out total loss by the 0.002496 match threshold.

Artifacts: `/data/chess/chess-engine-5/models/supervision-queue-2026.11.18/stratified-value-targets`.
