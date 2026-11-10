# Training batch size

Test whether batch size 64 or 256 improves the current baseline under the same training-time budget.

Starting baseline `/data/chess/chess-engine-5/models/training-recipe-experiments.VioD2w/one-layer`. All candidates started from fresh weights and used the baseline's one-layer architecture, 5,000-game dataset, and held-out split.
Baseline optimizer `{"beta1": 0.9, "beta2": 0.999, "eps": 1e-08, "kind": "adamw", "learning_rate": 0.001, "schedule": {"mode": "constant"}, "weight_decay": 0.01}` and batch size 128.

| Variant | Timed seconds | Optimizer steps | Policy loss | WDL loss |
|---|---:|---:|---:|---:|
| Control | 295.29 | 20,429 | 2.394013 | 0.857868 |
| 64 | 298.56 | 19,542 | 2.408915 | 0.860917 |
| 256 | 292.44 | 18,504 | 2.386716 | 0.856283 |

Screening required a policy loss improvement of at least 0.005, a WDL loss increase of at most 0.005, and lower combined loss.

Selected variant `256` for the gainer match. Candidate checkpoint `/data/chess/chess-engine-5/models/sequential-one-layer.YK4LRz/training-batch-size-256-run-1`.

## Baseline match

The inference benchmark assigned 800 visits per move to the baseline and 803 to the candidate. The gainer match was capped at 256 pairs.

- W/D/L 182/126/204
- Observed Elo -14.9
- SPRT verdict Inconclusive

Logs, checkpoints, and config snapshots are in `/data/chess/chess-engine-5/models/sequential-one-layer.YK4LRz`. Source revision `944157a`.
