# Cosine schedule

Test whether a cosine learning-rate schedule improves held-out prediction within five minutes.

| Learning-rate schedule | Timed seconds | Optimizer steps | Policy loss | WDL loss |
|---|---:|---:|---:|---:|
| Constant 0.001 (control) | 295.29 | 20,429 | 2.394013 | 0.857868 |
| Cosine 0.001 | 294.78 | 20,429 | 2.395501 | 0.862526 |
| Cosine 0.0015 | 306.01 | 20,429 | 2.388512 | 0.859059 |
| Cosine 0.002 | 314.45 | 20,429 | 2.385091 | 0.857634 |

Screening required a policy loss improvement of at least 0.005, a WDL loss increase of at most 0.005, and lower combined loss.
Cosine variants used 5% warmup and decayed to 10% of peak over the planned optimizer steps.

Selected variant `0.002` for the gainer match. Candidate checkpoint `/data/chess/chess-engine-5/models/sequential-one-layer.YK4LRz/cosine-schedule-0.002-run-1`.

## Match

- W/D/L 60/53/89
- Observed Elo -50.2
- SPRT verdict Failed

Logs, checkpoints, and config snapshots are in `/data/chess/chess-engine-5/models/sequential-one-layer.YK4LRz`. Source revision `944157a`.
