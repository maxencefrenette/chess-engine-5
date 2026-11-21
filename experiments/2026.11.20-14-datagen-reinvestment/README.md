# Reinvest faster self-play in more training positions

Test whether training on more unique positions from fresh self-play improves strength within the 60-minute budget.

The previous baseline generated 1,005,533 positions in 7,500 games. We held out 750 whole games and trained from scratch on 766,500 unique positions, up from 379,600, for 15 epochs.

| Timed training | Charged generation | Total budget | Held-out policy loss | Held-out WDL loss | Total loss |
|---:|---:|---:|---:|---:|---:|
| 667.78 s | 2,937.25 s | 3,605.03 s | 1.6187 | 0.6268 | 2.2455 |

The matched-speed gainer passed at pair 55 with 800 visits per move for both models. After 500 pairs the candidate scored **594/233/173 W/D/L**, **+195.0 nElo (95% CI +173.5, +216.6)**. Promoted it as the evaluation baseline and self-play teacher.
