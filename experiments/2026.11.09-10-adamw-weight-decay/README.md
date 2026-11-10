# Adamw weight decay

Test whether AdamW weight decay 0 or 0.05 improves the current baseline under the same training-time budget.

Starting baseline `/data/chess/chess-engine-5/models/training-recipe-experiments.VioD2w/one-layer`. All candidates started from fresh weights and used the baseline's one-layer architecture, 5,000-game dataset, and held-out split.
Baseline optimizer `{"beta1": 0.9, "beta2": 0.999, "eps": 1e-08, "kind": "adamw", "learning_rate": 0.001, "schedule": {"mode": "constant"}, "weight_decay": 0.01}` and batch size 128.

| Variant | Timed seconds | Optimizer steps | Policy loss | WDL loss |
|---|---:|---:|---:|---:|
| Control | 295.29 | 20,429 | 2.394013 | 0.857868 |
| 0 | 309.85 | 20,429 | 2.394775 | 0.856886 |
| 0.05 | 309.45 | 20,429 | 2.393616 | 0.862382 |

Screening required a policy loss improvement of at least 0.005, a WDL loss increase of at most 0.005, and lower combined loss.

No variant passed the held-out screen.

Logs, checkpoints, and config snapshots are in `/data/chess/chess-engine-5/models/sequential-one-layer.YK4LRz`. Source revision `944157a`.
