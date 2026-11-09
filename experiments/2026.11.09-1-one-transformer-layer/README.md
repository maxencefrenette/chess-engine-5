# One transformer layer

Test whether a one-layer policy transformer learns better under the same wall-clock budget.

Training override `model.layers=1`. All runs used dataset `/data/chess/chess-engine-5/training-data/v2/1794199969642733611-305557`, the same 4,500/500 game split, seed 42, batch size 128, and learning rate 0.001. Step counts were calibrated separately for the 285–315 second training window.

| Run | Timed seconds | Optimizer steps | Held-out policy loss | Held-out WDL loss |
|---|---:|---:|---:|---:|
| Five-minute control | 298.11 | 12,288 | 2.395401 | 0.860231 |
| Candidate | 295.29 | 20,429 | 2.394013 | 0.857868 |

Candidate checkpoint `/data/chess/chess-engine-5/models/architecture-ablations.sdLWM7/one-transformer-layer-run-1`. Logs, configs, and the experiment code diff from `09c29cf` are in `/data/chess/chess-engine-5/models/architecture-ablations.sdLWM7`.

## Screening

The one-layer model completed 66% more updates and improved both held-out losses. Its policy loss gain of 0.001388 fell below the 0.005 screening threshold. The experiment measured training throughput, so any inference advantage remains unmeasured.

The held-out screen required a policy loss improvement of at least 0.005, WDL loss within 0.005 of the control or better, and an improvement in combined loss.
