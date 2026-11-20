# Root WDL fraction

Compare fractions 0.25 and 0.75 against 0.5 by match. Held-out WDL uses each candidate’s own fraction.

| Candidate | Positions | Policy loss | Mixed WDL loss | Total loss | Training (s) | Generation (s) | Budget (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| fresh | 309,700 | 1.80908 | 0.68782 | 2.49690 | 356.21 | 3245.91 | 3602.12 |
| root-0.25 | 309,700 | 1.81145 | 0.68488 | 2.49633 | 356.25 | 3245.91 | 3602.16 |
| root-0.75 | 309,700 | 1.80494 | 0.69389 | 2.49882 | 359.28 | 3245.91 | 3605.19 |

| Candidate | Opponent | Visits (candidate/baseline) | W/D/L | nElo (95% CI) | Gainer SPRT |
|---|---|---:|---|---|---|
| root-0.25 | fresh | 800/800 | 656/656/652 | +4.1 (-12.1, +21.6) | failed |
| root-0.75 | fresh | 800/800 | 140/178/168 | -24.0 (-55.7, +8.1) | failed |

Kept the incumbent.
