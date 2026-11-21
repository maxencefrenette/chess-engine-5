# Sampled king-and-pawn WDL targets

Test whether cheap, uniquely sampled king-and-pawn versus king (KPK) positions improve endgame conversion and playing strength. Each arm trained for 15 epochs. KPK positions supplied only exact WDL targets.

| KPK positions | Self-play positions | Sampling and probing (s) | Timed training incl. sampling (s) | Self-play generation (s) | Budget (s) | Self-play held-out loss |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 779,600 | 0.00 | 675.66 | 2920.28 | 3595.94 | 2.15662 |
| 12,000 | 778,000 | 0.27 | 687.05 | 2914.29 | 3601.33 | 2.16112 |
| 30,000 | 774,200 | 0.41 | 693.76 | 2900.05 | 3593.81 | 2.17257 |

The 4,096 random test positions had a fresh halfmove clock and excluded training geometries, including file reflections and color reversals. From 128 of the winning starts, the engine played the pawn side at 800 visits and temperature zero against a defender choosing among WDL-optimal moves. The pawn side moved first in 64 starts. We selected by conversions, with self-play held-out loss as the tiebreaker.

| KPK positions | Correct WDL | Accuracy | KPK WDL loss | Conversion W/D/L |
|---:|---:|---:|---:|---:|
| 0 | 2,851/4,096 | 69.6% | 0.5501 | 124/4/0 |
| 12,000 | 3,648/4,096 | 89.1% | 0.2413 | 125/3/0 |
| 30,000 | 3,839/4,096 | 93.7% | 0.1464 | 126/2/0 |

On the paired starts, the 30,000-position candidate won three starts the baseline drew and drew one the baseline won. On ordinary held-out self-play, policy loss rose from 1.49447 to 1.50949 and WDL loss from 0.66215 to 0.66307.

The 30,000-position candidate **failed** the matched-speed gainer after 168 pairs at 800 visits per move. W/D/L **96/109/131**, nElo **-41.0 (-79.1, -2.6)** (95% CI). Kept the baseline.
