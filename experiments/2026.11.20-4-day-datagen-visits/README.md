# Self-play visits per move

Test how deeply to search during self-play at a 60-minute budget. All four datasets came from the same checkpoint, and each candidate trained for 15 epochs. The 800-visit arm generated 3,250 games to supply enough training positions after the game-level split. The other arms generated 2,500 games each.

| Candidate | Positions | Policy loss | Mixed WDL loss | Total loss | Training (s) | Generation (s) | Budget (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| visits-1000 | 303,100 | 1.70709 | 0.69882 | 2.40591 | 364.12 | 3235.52 | 3599.64 |
| visits-800 | 370,100 | 1.68436 | 0.69684 | 2.38120 | 440.67 | 3155.95 | 3596.63 |
| visits-1200 | 257,200 | 1.69970 | 0.63758 | 2.33728 | 309.64 | 3290.76 | 3600.40 |
| visits-1400 | 220,000 | 1.72912 | 0.66817 | 2.39730 | 263.01 | 3335.84 | 3598.85 |

| Held-out source | visits-1000 | visits-800 | visits-1200 | visits-1400 |
|---|---:|---:|---:|---:|
| visits-1000 | 2.40591 | 2.39115 | 2.42559 | 2.43421 |
| visits-800 | 2.39553 | 2.38120 | 2.41121 | 2.42203 |
| visits-1200 | 2.31230 | 2.30424 | 2.33728 | 2.34371 |
| visits-1400 | 2.37149 | 2.36376 | 2.39024 | 2.39730 |

| Candidate | Opponent | Visits (candidate/baseline) | W/D/L | nElo (95% CI) | Gainer SPRT |
|---|---|---:|---|---|---|
| visits-800 | previous baseline | 800/800 | 149/95/94 | +62.8 (+23.8, +101.5) | passed |

The 800-visit model had the lowest total loss on all four held-out sets. It passed the gainer against the previous baseline and was promoted.
