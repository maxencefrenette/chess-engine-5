# Learned piece–square embeddings

Test whether a learned vector for each piece and square improves playing strength. The residual embedding starts at zero.

| Candidate | Unique positions | Epochs | Policy loss | WDL loss | Total loss | Training (s) | Charged generation (s) | Budget (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 779,600 | 15 | 1.49447 | 0.66215 | 2.15662 | 675.66 | 2920.28 | 3595.94 |
| Piece–square | 779,600 | 15 | 1.49883 | 0.66429 | 2.16312 | 677.21 | 2920.28 | 3597.49 |

At batch 256, inference took 1.067 ms for the baseline and 1.094 ms for the candidate. The match used 800 and 781 visits per move, respectively.

| Candidate | Opponent | Pairs | W/D/L | nElo (95% CI) | Gainer |
|---|---|---:|---|---|---|
| Piece–square | Baseline | 427 | 265/302/287 | -7.6 (-31.7, +17.0) | failed |

Kept the baseline.
