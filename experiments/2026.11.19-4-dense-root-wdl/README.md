# Dense root WDL supervision

Supervise root WDL on every selected position and game outcomes on one eighth. Give each source half the value objective.

| Candidate | Positions | Policy | Outcome WDL | Total loss | Training (s) | Generation (s) | Budget (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| mixed-root-wdl | 309,000 | 1.83947 | 0.66112 | 2.50059 | 358.53 | 3228.22 | 3586.75 |
| dense-root-wdl | 309,000 | 1.83968 | 0.65972 | 2.49941 | 369.20 | 3228.22 | 3597.42 |

| Candidate | Opponent | SPRT | W/D/L | nElo (95% CI) | Verdict |
|---|---|---|---|---|---|
| dense-root-wdl | mixed-root-wdl | gainer | 466/564/472 | +1.4 (-17.0, +20.7) | failed |

Kept the incumbent.

Artifacts in `/data/chess/chess-engine-5/models/root-wdl-2026.11.18/dense-root-wdl`.
