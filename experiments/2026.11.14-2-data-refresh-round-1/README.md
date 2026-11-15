# Self-play data refresh, round 1

Test whether fresh self-play from the current baseline improves strength within the 60-minute budget.

Generated 432,427 positions from 2,500 games. Trained on 317,840 positions for 15 epochs.

| Timed training (s) | Generation charge (s) | Total budget (s) | Policy loss | WDL loss | Total loss |
|---:|---:|---:|---:|---:|---:|
| 314.22 | 3286.51 | 3600.73 | 1.9592 | 0.7056 | 2.6647 |

Gainer SPRT [0, 20] nElo **passed** after **73 game pairs**, at 800 visits per move for both models after matched-speed validation.

- W/D/L **76/46/24**
- nElo **+164.9 (+100.2, +229.4)**, 95% CI

Promoted the candidate; it will generate the next round’s data.

Artifacts: `/data/chess/chess-engine-5/models/data-crank-2026.11.14/round-01`.
