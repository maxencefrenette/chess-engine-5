# No smolgen

Test whether removing the Smolgen attention bias improves the model under the same wall-clock budget.

Training override `model.disable_smolgen=true`. Step counts were calibrated for the same training-time window.

| Run | Timed seconds | Optimizer steps | Held-out policy loss | Held-out WDL loss |
|---|---:|---:|---:|---:|
| Five-minute control | 298.11 | 12,288 | 2.395401 | 0.860231 |
| Candidate | 295.68 | 18,234 | 2.402215 | 0.858544 |

Candidate checkpoint `/data/chess/chess-engine-5/models/architecture-ablations.sdLWM7/no-smolgen-run-1`. Logs, configs, and the experiment code diff from `09c29cf` are in `/data/chess/chess-engine-5/models/architecture-ablations.sdLWM7`.

## Screening

Removing Smolgen allowed 48% more updates but increased held-out policy loss by 0.006814.

The held-out screen required a policy loss improvement of at least 0.005, WDL loss within 0.005 of the control or better, and an improvement in combined loss.
