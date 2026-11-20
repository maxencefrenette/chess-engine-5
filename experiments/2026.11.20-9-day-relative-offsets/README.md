# Relative square offsets

Learn a 15 × 15 table of relative-square attention biases per head and layer.

| Candidate | Positions | Policy loss | Mixed WDL loss | Total loss | Training (s) | Generation (s) | Budget (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| No offset table | 383,300 | 1.73799 | 0.68796 | 2.42595 | 330.97 | 3268.51 | 3599.48 |
| Learned offsets | 383,300 | 1.73526 | 0.68771 | 2.42298 | 366.22 | 3268.51 | 3634.73 |

| Candidate | Opponent | Visits (candidate/baseline) | W/D/L | nElo (95% CI) | Gainer SPRT |
|---|---|---:|---|---|---|
| Learned offsets | No offset table | 698/800 | 551/415/558 | +1.4 (-17.0, +20.7) | failed |

At batch 512, the offset table increased inference time from 2.257 to 2.587 ms. Kept the side-to-move model without offsets.
