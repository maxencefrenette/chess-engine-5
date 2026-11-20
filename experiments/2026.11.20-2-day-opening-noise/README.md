# Opening root noise in self-play

Mix 0%, 10%, or 25% Dirichlet noise into root priors for the first 30 plies, with concentration 0.3. Generate each dataset from the same checkpoint. Select the alternative with the lowest mean total loss across the common held-out sets.

| Candidate | Positions | Policy loss | Mixed WDL loss | Total loss | Training (s) | Generation (s) | Budget (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| noise-0 | 301,700 | 1.72909 | 0.63380 | 2.36289 | 360.94 | 3234.22 | 3595.16 |
| noise-10 | 300,700 | 1.74274 | 0.62704 | 2.36978 | 359.53 | 3235.96 | 3595.49 |
| noise-25 | 300,000 | 1.79004 | 0.65552 | 2.44555 | 360.09 | 3236.45 | 3596.53 |

| Held-out source | noise-0 | noise-10 | noise-25 | Distinct ply-16 positions / games reaching ply 16 |
|---|---:|---:|---:|---:|
| noise-0 | 2.36289 | 2.37736 | 2.37490 | 2471/2499 |
| noise-10 | 2.36592 | 2.36978 | 2.37149 | 2486/2496 |
| noise-25 | 2.45663 | 2.45235 | 2.44555 | 2490/2490 |

| Candidate | Opponent | Visits (candidate/baseline) | W/D/L | nElo (95% CI) | Gainer SPRT |
|---|---|---:|---|---|---|
| noise-25 | cool-45 | 800/800 | 826/630/816 | +5.1 (-10.4, +21.8) | failed |

Kept the incumbent.
