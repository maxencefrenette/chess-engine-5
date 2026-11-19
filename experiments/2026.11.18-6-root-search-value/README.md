# Search value as auxiliary supervision

Generate fresh self-play with the incumbent. On sampled WDL rows, add squared error between predicted win minus loss and the root’s backed-up score. The coefficient is relative to game-result cross-entropy. All arms trained on the same 314,000 paid positions.

| Setting | Policy | WDL | Total loss | Runtime (s) | Generation (s) | Total budget (s) |
|---|---:|---:|---:|---:|---:|---:|
| game-result-only | 1.911006 | 0.742958 | 2.653964 | 301.23 | 3295.76 | 3596.99 |
| q-0.25 | 1.874025 | 0.742413 | 2.616438 | 321.13 | 3295.76 | 3616.89 |
| q-1.0 | 1.872659 | 0.740259 | 2.612918 | 318.47 | 3295.76 | 3614.23 |

Common held-out total loss (policy + WDL):

| Position source | Incumbent | game-result-only | q-0.25 | q-1.0 |
|---|---:|---:|---:|---:|
| Incumbent data | 2.629102 | 2.696257 | 2.675161 | 2.670660 |
| Fresh data | 2.624374 | 2.653964 | 2.616438 | 2.612918 |

Selected q-1.0. Matched-speed gainer passed after 193 game pairs. W/D/L 142/149/95. Observed nElo (95% CI) +53.0 (+17.4, +88.3).

Promoted the candidate.

Artifacts: `/data/chess/chess-engine-5/models/supervision-queue-2026.11.18/root-search-value`.
