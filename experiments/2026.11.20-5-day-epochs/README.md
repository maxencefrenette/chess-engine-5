# Epochs versus unique positions

Trade epochs for unique 800-visit positions at a 60-minute budget.

| Candidate | Positions | Policy loss | Mixed WDL loss | Total loss | Training (s) | Generation (s) | Budget (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4 epochs | 407,000 | 1.83160 | 0.69411 | 2.52571 | 129.02 | 3470.59 | 3599.62 |
| 8 epochs | 392,900 | 1.70301 | 0.69126 | 2.39427 | 252.74 | 3350.37 | 3603.10 |
| 12 epochs | 379,800 | 1.68730 | 0.69016 | 2.37747 | 359.05 | 3238.66 | 3597.71 |
| 15 epochs | 370,100 | 1.68436 | 0.69684 | 2.38120 | 440.67 | 3155.95 | 3596.63 |
| 20 epochs | 355,900 | 1.68145 | 0.68663 | 2.36808 | 573.62 | 3034.88 | 3608.50 |

The 15-epoch arm is the incumbent checkpoint. All arms used the same 325 held-out games. The finalist was the alternative with the lowest held-out total loss.

| Candidate | Opponent | Visits (candidate/baseline) | W/D/L | nElo (95% CI) | Gainer SPRT |
|---|---|---:|---|---|---|
| 20 epochs | visits-800 | 800/800 | 1126/898/1094 | +7.3 (-6.2, +22.2) | failed |

Kept the incumbent.
