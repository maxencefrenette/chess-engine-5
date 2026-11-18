# Train on original and reflected positions

Test whether spending less on self-play and training on both views of positions without castling rights improves held-out loss within 60 minutes. Each epoch shuffles all originals and one reflection per eligible original. The 9-epoch follow-up spends the training savings on more source positions. Both candidates were scored on the baseline’s 45,045 unreflected held-out positions.

| Recipe | Source positions | Examples/epoch | Epochs | Policy loss | WDL loss | Total loss | Training runtime (s) | Generation (s) | Budget (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 317,840 | 317,840 | 15 | 1.938736 | 0.690366 | 2.629102 | 309.04 | 3286.51 | 3595.55 |
| Both views | 301,200 | 503,112 | 15 | 1.952555 | 0.698856 | 2.651411 | 484.64 | 3114.46 | 3599.10 |
| Both views | 317,800 | 530,836 | 9 | 1.953711 | 0.686924 | 2.640635 | 305.70 | 3286.10 | 3591.80 |

Neither candidate cleared the held-out loss threshold for a matched-speed gainer.

Artifacts: `/data/chess/chess-engine-5/models/both-views-2026.11.18`.
