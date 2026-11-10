# AdamW weight decay

Test whether AdamW weight decay 0 or 0.05 improves held-out prediction under the same training-time budget.

| Variant | Timed seconds | Optimizer steps | Policy loss | WDL loss |
|---|---:|---:|---:|---:|
| 0.01 (control) | 295.29 | 20,429 | 2.394013 | 0.857868 |
| 0 | 309.85 | 20,429 | 2.394775 | 0.856886 |
| 0.05 | 309.45 | 20,429 | 2.393616 | 0.862382 |

Screening required a policy loss improvement of at least 0.005, a WDL loss increase of at most 0.005, and lower combined loss.

No variant passed the held-out screen.

Logs, checkpoints, and config snapshots are in `/data/chess/chess-engine-5/models/sequential-one-layer.YK4LRz`. Source revision `944157a`.
