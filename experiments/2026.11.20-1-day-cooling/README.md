# Self-play cooling schedule

Lower the sampling temperature from 1 to 0.5 at ply 15, 30, or 45. Generate each dataset from the same checkpoint. Select the alternative with the lowest mean total loss across the common held-out sets.

| Candidate | Positions | Policy loss | Mixed WDL loss | Total loss | Training (s) | Generation (s) | Budget (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| cool-30 | 307,300 | 1.75446 | 0.66038 | 2.41484 | 366.11 | 3231.12 | 3597.23 |
| cool-15 | 310,100 | 1.78447 | 0.71773 | 2.50220 | 371.55 | 3227.92 | 3599.47 |
| cool-45 | 304,800 | 1.73590 | 0.65064 | 2.38654 | 369.15 | 3234.93 | 3604.08 |

| Held-out source | cool-30 | cool-15 | cool-45 |
|---|---:|---:|---:|
| cool-30 | 2.41484 | 2.42331 | 2.42212 |
| cool-15 | 2.50568 | 2.50220 | 2.50030 |
| cool-45 | 2.39134 | 2.39704 | 2.38654 |

| Candidate | Opponent | Visits (candidate/baseline) | W/D/L | nElo (95% CI) | Gainer SPRT |
|---|---|---:|---|---|---|
| cool-45 | attention-16 | 800/800 | 237/217/180 | +35.5 (+7.0, +63.5) | passed |

Promoted cool-45.
