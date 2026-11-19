# Mix search WDL and game outcomes

Train on equal parts game outcome and backed-up root WDL on sampled rows.

| Candidate | Positions | Policy | Outcome WDL | Total loss | Training (s) | Generation (s) | Budget (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| generation-1 | 309,000 | 1.83968 | 0.66024 | 2.49992 | 373.67 | 3228.22 | 3601.89 |
| mixed-root-wdl | 309,000 | 1.83947 | 0.66112 | 2.50059 | 358.53 | 3228.22 | 3586.75 |

| Candidate | Opponent | SPRT | W/D/L | nElo (95% CI) | Verdict |
|---|---|---|---|---|---|
| mixed-root-wdl | generation-1 | non-regression | 289/292/267 | +7.9 (-16.9, +32.0) | passed |

Promoted mixed-root-wdl.

Artifacts in `/data/chess/chess-engine-5/models/root-wdl-2026.11.18/mixed-root-wdl`.
