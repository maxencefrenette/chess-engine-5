# Self-play data refresh, round 2

Test whether fresh self-play from the current baseline improves strength within the 60-minute budget.

Generated 443,799 positions from 2,500 games. Trained on 313,041 positions for 15 epochs.

| Timed training (s) | Generation charge (s) | Total budget (s) | Policy loss | WDL loss | Total loss |
|---:|---:|---:|---:|---:|---:|
| 309.04 | 3289.93 | 3598.98 | 1.9018 | 0.7429 | 2.6447 |

Gainer SPRT [0, 20] nElo **failed** after **759 game pairs**, at 800 visits per move for both models after matched-speed validation.

- W/D/L **448/616/454**
- nElo **+1.4 (-17.0, +20.7)**, 95% CI

Kept the previous baseline and stopped the data-refresh loop.

Artifacts: `/data/chess/chess-engine-5/models/data-crank-2026.11.14/round-02`.
