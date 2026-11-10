# Cosine schedule

Test whether a cosine learning-rate schedule improves the one-layer baseline within five minutes.

Starting baseline `/data/chess/chess-engine-5/models/training-recipe-experiments.VioD2w/one-layer`. All candidates started from fresh weights and used the baseline's one-layer architecture, 5,000-game dataset, and held-out split.
Baseline optimizer `{"beta1": 0.9, "beta2": 0.999, "eps": 1e-08, "kind": "adamw", "learning_rate": 0.001, "schedule": {"mode": "constant"}, "weight_decay": 0.01}` and batch size 128.

| Variant | Timed seconds | Optimizer steps | Policy loss | WDL loss |
|---|---:|---:|---:|---:|
| Control | 295.29 | 20,429 | 2.394013 | 0.857868 |
| 0.001 | 294.78 | 20,429 | 2.395501 | 0.862526 |
| 0.0015 | 306.01 | 20,429 | 2.388512 | 0.859059 |
| 0.002 | 314.45 | 20,429 | 2.385091 | 0.857634 |

Screening required a policy loss improvement of at least 0.005, a WDL loss increase of at most 0.005, and lower combined loss.
Cosine variants used 5% warmup and decayed to 10% of peak over the planned optimizer steps.

Selected variant `0.002` for the gainer match. Candidate checkpoint `/data/chess/chess-engine-5/models/sequential-one-layer.YK4LRz/cosine-schedule-0.002-run-1`.

## Baseline match

The inference benchmark assigned 800 visits per move to the baseline and 799 to the candidate. The gainer match was capped at 256 pairs.

- W/D/L 60/53/89
- Observed Elo -50.2
- SPRT verdict Failed

Logs, checkpoints, and config snapshots are in `/data/chess/chess-engine-5/models/sequential-one-layer.YK4LRz`. Source revision `944157a`.
