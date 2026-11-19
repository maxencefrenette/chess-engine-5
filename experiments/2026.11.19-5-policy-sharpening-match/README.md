# Policy sharpening in a match

Test the saved exponent-1.25 checkpoint against its original incumbent.

| Candidate | Positions | Policy | Outcome WDL | Total loss | Training (s) | Generation (s) | Budget (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| exponent-1.25 | 317,840 | 1.95748 | 0.69326 | 2.65074 | 306.09 | 3286.51 | 3592.60 |

| Candidate | Opponent | SPRT | W/D/L | nElo (95% CI) | Verdict |
|---|---|---|---|---|---|
| exponent-1.25 | cosine-002-no-hints | gainer | 89/127/124 | -40.3 (-78.2, -2.2) | failed |

Kept the incumbent.

Artifacts in `/data/chess/chess-engine-5/models/root-wdl-2026.11.18/policy-sharpening-match`.
