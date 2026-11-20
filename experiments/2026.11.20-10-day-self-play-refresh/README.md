# Fresh self-play

Test whether refreshing self-play with the promoted side-to-move model improves playing strength.

The teacher generated 446,846 positions from 3,250 games at 800 visits per move. The candidate trained from scratch for 15 epochs, with 325 whole games held out.

| Candidate | Positions | Policy loss | Mixed WDL loss | Total loss | Training (s) | Generation (s) | Budget (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fresh self-play | 379,600 | 1.67412 | 0.63602 | 2.31014 | 327.81 | 3272.12 | 3599.92 |

| Candidate | Opponent | Visits (candidate/baseline) | W/D/L | nElo (95% CI) | Gainer SPRT |
|---|---|---:|---|---|---|
| Fresh self-play | Self-play teacher | 800/800 | 493/227/280 | +88.4 (+66.9, +110.0) | passed |

The gainer passed at pair 133. Promoted the candidate after 500 pairs.
