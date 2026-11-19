# Isolate the effect of root supervision

Compare the promoted root-value checkpoint directly with the outcome-only control trained on the identical 314,000 positions. Both checkpoints come from the root-value sweep.

| Model | Timed runtime (s) | Generation (s) | Total budget (s) |
|---|---:|---:|---:|
| Outcome only | 301.23 | 3295.76 | 3596.99 |
| Root supervision | 318.47 | 3295.76 | 3614.23 |

Matched-speed gainer passed after 235 game pairs. W/D/L 177/168/125. Observed nElo (95% CI) +46.2 (+13.2, +78.9).

Artifacts: `/data/chess/chess-engine-5/models/supervision-queue-2026.11.18/isolate-root-supervision`.
