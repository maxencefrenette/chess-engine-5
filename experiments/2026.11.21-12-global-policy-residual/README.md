# Global policy correction after attention pooling

Test whether a 128-wide global decoder improves the baseline’s from/to move scores. Two attention stages pool the 64-wide square features before projecting each pooled vector to 128 dimensions. The decoder adds a learned correction per move, initialized to zero.

| Policy head | Parameters (whole model) | Held-out policy loss | WDL loss | Total loss | Timed training (s) | Charged generation (s) | Budget (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 87,843 | 1.49447 | 0.66215 | 2.15662 | 675.66 | 2920.28 | 3595.94 |
| Baseline + global decoder | 430,627 | 1.49854 | 0.66101 | 2.15954 | 848.53 | 2920.28 | 3768.81 |

Switching off the trained correction raised policy loss to 1.52213 on the same 96,290 held-out positions. At batch 256, inference took 1.231 ms for the candidate and 1.067 ms for the baseline, giving them 693 and 800 visits per move respectively.

The [0, 15] nElo gainer **failed** after 147 game pairs. Candidate W/D/L **71/99/124**, estimated **−75.4 nElo (−116.5, −34.2)**, 95% CI. Kept the baseline.
