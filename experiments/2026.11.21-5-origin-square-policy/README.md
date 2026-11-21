# Policy supervision by origin square

Test whether an extra loss on which piece to move improves the move policy. Both the search target and model probabilities are summed by origin square. We selected the auxiliary weight by held-out policy plus WDL loss.

| Candidate | Unique positions | Epochs | Policy loss | WDL loss | Total loss | Training (s) | Charged generation (s) | Budget (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 779,600 | 15 | 1.49447 | 0.66215 | 2.15662 | 675.66 | 2920.28 | 3595.94 |
| 0.1 | 779,600 | 15 | 1.49641 | 0.66127 | 2.15768 | 751.83 | 2920.28 | 3672.11 |
| 0.3 | 779,600 | 15 | 1.50997 | 0.66368 | 2.17366 | 749.89 | 2920.28 | 3670.17 |

| Candidate | Opponent | Pairs | W/D/L | nElo (95% CI) | Gainer |
|---|---|---:|---|---|---|
| 0.1 | Baseline | 561 | 376/354/392 | -2.7 (-23.8, +19.2) | failed |

Kept the baseline.
