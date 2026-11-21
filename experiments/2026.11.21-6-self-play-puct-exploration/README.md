# Self-play PUCT exploration

Test whether changing the exploration constant used to generate self-play improves the next model. All three datasets used the same teacher and 800 visits per move. Evaluation used exploration 1.4 for both players.

| Candidate | Unique positions | Epochs | Policy loss | WDL loss | Total loss | Training (s) | Charged generation (s) | Budget (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 779,600 | 15 | 1.49447 | 0.66215 | 2.15662 | 675.66 | 2920.28 | 3595.94 |
| 1.0 | 781,100 | 15 | 1.54054 | 0.65933 | 2.19988 | 673.57 | 2922.89 | 3596.46 |
| 2.0 | 775,600 | 15 | 1.49329 | 0.68051 | 2.17380 | 667.17 | 2927.67 | 3594.84 |

Held-out total loss on the same three sets of games. The mean gives each search setting equal weight.

| Training exploration | 1.4 holdout | 1.0 holdout | 2.0 holdout | Mean |
|---:|---:|---:|---:|---:|
| 1.4 | 2.15662 | 2.18822 | 2.16764 | 2.17083 |
| 1.0 | 2.17845 | 2.19988 | 2.19290 | 2.19041 |
| 2.0 | 2.17148 | 2.20794 | 2.17380 | 2.18441 |

| Candidate | Opponent | Pairs | W/D/L | nElo (95% CI) | Gainer |
|---|---|---:|---|---|---|
| 2.0 | Baseline | 185 | 103/131/136 | -36.5 (-73.1, +0.3) | failed |

Kept the baseline.
