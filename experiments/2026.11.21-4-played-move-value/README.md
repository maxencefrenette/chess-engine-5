# Predicting the value of the played move

Test whether predicting the next search’s WDL from the parent board and played move improves the deployed model. Only consecutive selected positions supplied a target, expressed from the parent’s side to move. We selected the auxiliary weight by held-out policy plus WDL loss.

| Candidate | Unique positions | Epochs | Policy loss | WDL loss | Total loss | Training (s) | Charged generation (s) | Budget (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 779,600 | 15 | 1.49447 | 0.66215 | 2.15662 | 675.66 | 2920.28 | 3595.94 |
| 0.05 | 779,600 | 15 | 1.49585 | 0.66098 | 2.15683 | 720.34 | 2920.28 | 3640.62 |
| 0.15 | 779,600 | 15 | 1.49627 | 0.66035 | 2.15662 | 708.29 | 2920.28 | 3628.57 |

| Candidate | Opponent | Pairs | W/D/L | nElo (95% CI) | Gainer |
|---|---|---:|---|---|---|
| 0.15 | Baseline | 218 | 131/141/164 | -29.7 (-63.7, +4.6) | failed |

Kept the baseline.
