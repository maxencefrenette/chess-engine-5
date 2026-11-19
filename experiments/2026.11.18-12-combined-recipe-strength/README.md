# Combined recipe strength

Measure the final promoted checkpoint directly against the checkpoint at the start of this series. The final recipe uses root-value supervision, self-play cooling after ply 30, and a training-only opponent-reply head. The fixed match used seed 43, 1,000 game pairs, and 800 visits per move for both models after matched-speed verification.

| Model | Timed runtime (s) | Generation (s) | Total budget (s) |
|---|---:|---:|---:|
| Starting baseline | 309.04 | 3286.51 | 3595.55 |
| Final baseline | 383.56 | 3277.43 | 3660.99 |

W/D/L 1011/598/391. Observed nElo (95% CI) +141.8 (+126.6, +157.0).

Artifacts: `/data/chess/chess-engine-5/models/supervision-queue-2026.11.18/combined-recipe-strength`.
