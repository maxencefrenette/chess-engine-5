# Random value target masking

Test whether using fewer game-result targets per game helps a transformer WDL head generalize.

Randomly selected positions within each game once, using the same mask across epochs. WDL loss was averaged over active targets and measured on all held-out positions.

| Value sample rate | Value labels | Value games | p90 labels/game | Policy loss | WDL loss |
|---:|---:|---:|---:|---:|---:|
| 1.0 (prior run) | 250,301 | 1,800 | — | 2.3198 | 2.4000 |
| 0.5 | 125,617 | 1,800 | 116 | 2.3288 | 1.8005 |
| 0.25 | 62,794 | 1,800 | 58 | 2.3279 | 1.6024 |
| 0.125 | 31,406 | 1,800 | 29 | 2.3445 | 1.5609 |

Best masked rate by held-out WDL loss was 0.125. This incorrectly used the Transformer WDL head instead of the MLP baseline, so it failed to pass the SPRT test.

## Gainer SPRT

- W/D/L 1/37/62 over 100 games
- Observed Elo -246.3
- SPRT verdict Failed
- Visits per move 804 candidate, 800 baseline

Run artifacts in `/data/chess/chess-engine-5/models/supervision-2026.11.10.c0H3Yd/mask`.
