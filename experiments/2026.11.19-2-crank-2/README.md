# Self-play refresh 2

Train on fresh self-play from generation-1. Generated 400,085 positions across 2,500 games.

| Candidate | Positions | Policy | Outcome WDL | Total loss | Training (s) | Generation (s) | Budget (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| generation-2 | 310,800 | 1.79287 | 0.70658 | 2.49945 | 376.43 | 3224.60 | 3601.03 |

| Candidate | Opponent | SPRT | W/D/L | nElo (95% CI) | Verdict |
|---|---|---|---|---|---|
| generation-2 | generation-1 | gainer | 580/607/581 | +3.0 (-14.2, +21.2) | failed |

Kept the incumbent.

Artifacts in `/data/chess/chess-engine-5/models/root-wdl-2026.11.18/crank-2`.
