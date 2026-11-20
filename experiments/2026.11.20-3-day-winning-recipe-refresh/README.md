# Refresh self-play with the winning recipe

Test whether refreshing self-play data improves the winning recipe. The opening-noise control had already generated 386,688 positions in 2,500 games using the current baseline and its standard settings. The candidate trained on 301,700 of those positions.

| Candidate | Positions | Policy loss | Mixed WDL loss | Total loss | Training (s) | Generation (s) | Budget (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| fresh | 301,700 | 1.72909 | 0.63380 | 2.36289 | 362.08 | 3234.22 | 3596.30 |

| Candidate | Opponent | Visits (candidate/baseline) | W/D/L | nElo (95% CI) | Gainer SPRT |
|---|---|---:|---|---|---|
| fresh | previous baseline | 800/800 | 350/267/285 | +26.8 (+2.6, +50.4) | passed |

Promoted fresh.
