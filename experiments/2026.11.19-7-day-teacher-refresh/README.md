# Self-play refresh

Test whether refreshing self-play data improves playing strength. The baseline generated 397,949 positions in 2,500 games, and the candidate trained on 309,700 of them.

| Candidate | Positions | Policy loss | Mixed WDL loss | Total loss | Training (s) | Generation (s) | Budget (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| fresh | 309,700 | 1.80908 | 0.68782 | 2.49690 | 356.21 | 3245.91 | 3602.12 |

| Candidate | Opponent | Visits (candidate/baseline) | W/D/L | nElo (95% CI) | Gainer SPRT |
|---|---|---:|---|---|---|
| fresh | previous baseline | 800/800 | 125/93/80 | +70.2 (+28.8, +111.4) | passed |

Promoted fresh.
