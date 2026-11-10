# Piece-count MLP value masking

Test whether the piece-count WDL MLP benefits from fewer outcome labels per game.

Used a fixed random sample within each training game and dense policy supervision. Evaluated WDL on all held-out positions with the same split and 30-minute weighted budget.

| Value sample rate | Value labels | Value games | p90 labels/game | Policy loss | WDL loss |
|---:|---:|---:|---:|---:|---:|
| 1.0 (baseline) | 250,301 | 1,800 | — | 2.2945 | 0.8192 |
| 0.5 | 125,617 | 1,800 | 116 | 2.2923 | 0.8221 |
| 0.25 | 62,794 | 1,800 | 58 | 2.2930 | 0.8259 |

Selected 0.5 by held-out WDL loss among the masked arms.
The 0.25 rate did not beat both 0.5 and the unmasked control on held-out WDL loss, so 0.125 was skipped.
The best masked arm did not improve on baseline held-out WDL loss.

Run artifacts in `/data/chess/chess-engine-5/models/supervision-2026.11.10.c0H3Yd/mlp-value-masking`.
