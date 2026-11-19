# Cool self-play after ply 30

Generate self-play at temperature 1 for the first 30 plies and 0.5 thereafter. The match measures playing strength against the incumbent across the change in training distribution.

Losses in the first table use each model’s own held-out self-play games.

| Setting | Policy | WDL | Total loss | Runtime (s) | Generation (s) | Total budget (s) |
|---|---:|---:|---:|---:|---:|---:|
| temperature-1 | 1.786581 | 0.585360 | 2.371941 | 321.57 | 3283.06 | 3604.63 |
| cool-after-30 | 1.890616 | 0.644625 | 2.535242 | 327.85 | 3277.43 | 3605.28 |

Common held-out total loss (policy + WDL):

| Position source | Incumbent | temperature-1 | cool-after-30 |
|---|---:|---:|---:|
| Incumbent data | 2.612918 | 2.656347 | 2.713369 |
| Cooled data | 2.543547 | 2.552720 | 2.535242 |
| Temperature-1 data | 2.356465 | 2.371941 | 2.398680 |

Selected cool-after-30. Matched-speed gainer passed after 225 game pairs. W/D/L 161/175/114. Observed nElo (95% CI) +47.1 (+13.7, +80.1).

Promoted the candidate.

Artifacts: `/data/chess/chess-engine-5/models/supervision-queue-2026.11.18/cooling-self-play`.
