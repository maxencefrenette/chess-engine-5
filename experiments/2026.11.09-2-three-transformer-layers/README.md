# Three transformer layers

Test whether a third policy transformer layer improves held-out prediction under the same wall-clock budget.

Training override `model.layers=3`. All runs used dataset `/data/chess/chess-engine-5/training-data/v2/1794199969642733611-305557`, the same 4,500/500 game split, seed 42, batch size 128, and learning rate 0.001. Step counts were calibrated separately for the 285–315 second training window.

| Run | Timed seconds | Optimizer steps | Held-out policy loss | Held-out WDL loss |
|---|---:|---:|---:|---:|
| Five-minute control | 298.11 | 12,288 | 2.395401 | 0.860231 |
| Candidate | 290.79 | 8,720 | 2.406456 | 0.864709 |

Candidate checkpoint `/data/chess/chess-engine-5/models/architecture-ablations.sdLWM7/three-transformer-layers-run-1`. Logs, configs, and the experiment code diff from `09c29cf` are in `/data/chess/chess-engine-5/models/architecture-ablations.sdLWM7`.

## Screening

The third layer reduced the number of updates by 29% and worsened both held-out losses under the time budget.

The held-out screen required a policy loss improvement of at least 0.005, WDL loss within 0.005 of the control or better, and an improvement in combined loss.
