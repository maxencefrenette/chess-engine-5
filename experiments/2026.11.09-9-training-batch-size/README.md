# Training batch size

Test whether batch size 64 or 256 improves held-out prediction under the same training-time budget.

| Variant | Timed seconds | Optimizer steps | Policy loss | WDL loss |
|---|---:|---:|---:|---:|
| 128 (control) | 295.29 | 20,429 | 2.394013 | 0.857868 |
| 64 | 298.56 | 19,542 | 2.408915 | 0.860917 |
| 256 | 292.44 | 18,504 | 2.386716 | 0.856283 |

Screening required a policy loss improvement of at least 0.005, a WDL loss increase of at most 0.005, and lower combined loss.

Selected variant `256` for the gainer match. Candidate checkpoint `/data/chess/chess-engine-5/models/sequential-one-layer.YK4LRz/training-batch-size-256-run-1`.

## Match

- W/D/L 182/126/204
- Observed Elo -14.9
- SPRT verdict Inconclusive

Logs, checkpoints, and config snapshots are in `/data/chess/chess-engine-5/models/sequential-one-layer.YK4LRz`. Source revision `944157a`.
