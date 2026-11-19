# Self-play refresh 1

Train on fresh self-play from reply-0.15. Generated 426,769 positions across 2,500 games.

| Candidate | Positions | Policy | Outcome WDL | Total loss | Training (s) | Generation (s) | Budget (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| generation-1 | 309,000 | 1.83968 | 0.66024 | 2.49992 | 373.67 | 3228.22 | 3601.89 |

| Candidate | Opponent | SPRT | W/D/L | nElo (95% CI) | Verdict |
|---|---|---|---|---|---|
| generation-1 | reply-0.15 | gainer | 1568/1735/1421 | +11.2 (-2.4, +23.5) | passed |

Promoted generation-1.

Artifacts in `/data/chess/chess-engine-5/models/root-wdl-2026.11.18/crank-1`.
