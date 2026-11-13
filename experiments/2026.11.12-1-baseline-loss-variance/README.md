# Baseline loss variability

Measure variation in final held-out total loss across 10 baseline training runs.

| Seed | Policy loss | WDL loss | Total loss |
|---:|---:|---:|---:|
| 42 | 2.204167 | 0.757575 | 2.961742 |
| 43 | 2.214876 | 0.753340 | 2.968216 |
| 44 | 2.208193 | 0.759353 | 2.967547 |
| 45 | 2.206338 | 0.762996 | 2.969334 |
| 46 | 2.207157 | 0.757456 | 2.964613 |
| 47 | 2.207490 | 0.756733 | 2.964223 |
| 48 | 2.209762 | 0.753047 | 2.962809 |
| 49 | 2.211552 | 0.755434 | 2.966987 |
| 50 | 2.203111 | 0.761307 | 2.964418 |
| 51 | 2.209784 | 0.757485 | 2.967269 |

Mean total loss: **2.965716**. Sample standard deviation: **0.002496**, stored as `loss_stddev` in `config/eval-model.toml`.

Artifacts: `/data/chess/chess-engine-5/models/loss-variance-2026.11.12`.
