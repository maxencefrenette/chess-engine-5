# Isolate self-play cooling

Challenge the cooled baseline using temperature 1 throughout self-play. Both datasets were generated from the same teacher with identical seeds and opening conditions. Each model is charged for its own selected training positions.

Both arms used root-value weight 1.0, reply weight 0.15 and 15 epochs.

| Total loss on held-out positions | Cooled | Temperature 1 |
|---|---:|---:|
| Cooled | 2.516616 | 2.549523 |
| Temperature 1 | 2.379590 | 2.374895 |

| Training data | Source positions | Runtime (s) | Generation (s) | Budget (s) |
|---|---:|---:|---:|---:|
| Cooled | 318,400 | 383.56 | 3277.43 | 3660.99 |
| Temperature 1 | 312,400 | 376.42 | 3285.16 | 3661.58 |

The temperature-1 challenger failed its matched-speed gainer after 152 game pairs. W/D/L 79/112/113. Observed nElo (95% CI) -47.7 (-88.0, -7.1).

Artifacts: `/data/chess/chess-engine-5/models/supervision-queue-2026.11.18/temperature-control-match`.
