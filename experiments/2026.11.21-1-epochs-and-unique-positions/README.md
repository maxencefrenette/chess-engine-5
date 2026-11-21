# Epochs versus unique training positions

Trade epochs for unique positions under a 60-minute budget on the same shards and held-out games. The 15-epoch control is the incumbent checkpoint.

| Candidate | Unique positions | Epochs | Policy loss | WDL loss | Total loss | Training (s) | Charged generation (s) | Budget (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 epochs | 832,600 | 10 | 1.51519 | 0.66510 | 2.18029 | 480.62 | 3118.81 | 3599.43 |
| 15 epochs | 779,600 | 15 | 1.49447 | 0.66215 | 2.15662 | 675.66 | 2920.28 | 3595.94 |
| 20 epochs | 734,400 | 20 | 1.50613 | 0.66137 | 2.16750 | 844.10 | 2750.96 | 3595.06 |

The 20-epoch candidate had the lowest held-out total loss among the alternatives.

| Candidate | Opponent | Pairs | W/D/L | nElo (95% CI) | Gainer |
|---|---|---:|---|---|---|
| 20 epochs | 15 epochs | 198 | 113/137/146 | -32.4 (-67.5, +2.9) | failed |

Kept 15 epochs.
