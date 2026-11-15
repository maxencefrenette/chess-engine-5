# Value supervision sweep

Tune random value-target sampling and value-loss weight within the 60-minute budget.

Selected the lowest held-out total loss separately within each weight, then ran one gainer per weight against the frozen baseline. Total loss is policy + WDL.

| Value weight | Sample rate | Policy loss | WDL loss | Total loss |
|---:|---:|---:|---:|---:|
| 0.03 | 1/16 | 1.958344 | 0.700817 | **2.659161** |
| 0.03 | 1/8 | 1.959161 | 0.705553 | 2.664713 |
| 0.03 | 1/4 | 1.958103 | 0.705391 | 2.663493 |
| 0.1 | 1/16 | 1.959876 | 0.700716 | **2.660591** |
| 0.1 | 1/8 | 1.960670 | 0.706265 | 2.666935 |
| 0.1 | 1/4 | 1.960606 | 0.704822 | 2.665429 |
| 0.3 | 1/16 | 1.972631 | 0.699965 | **2.672596** |
| 0.3 | 1/8 | 1.970604 | 0.707744 | 2.678348 |
| 0.3 | 1/4 | 1.975849 | 0.709358 | 2.685207 |

Timed training ranged from 307.64–315.44s; generation charge was 3,286.51s per run, giving total budgets of 3,594.15–3,601.95s. [Full measurements](results.csv).

All three weights selected **1/16 sampling**. Gainer SPRT [0, 20] nElo used 800 visits per move for both models after matched-speed validation.

| Value weight | Game pairs | W/D/L | nElo (95% CI) | Verdict |
|---:|---:|---|---|---|
| 0.03 | 114 | 48/99/81 | -68.6 (-115.2, -21.8) | Failed |
| 0.1 | 213 | 115/163/148 | -29.7 (-63.8, +4.6) | Failed |
| 0.3 | 112 | 51/86/87 | -71.0 (-118.2, -23.5) | Failed |

No candidate passed; kept the existing baseline.

Artifacts: `/data/chess/chess-engine-5/models/value-supervision-2026.11.14`.
