# Wider transformer

Test whether widening the policy transformer from 64 to 96 channels improves held-out prediction under the same wall-clock budget.

Training override `model.width=96`. Step counts were calibrated for the same training-time window.

| Run | Timed seconds | Optimizer steps | Held-out policy loss | Held-out WDL loss |
|---|---:|---:|---:|---:|
| Five-minute control | 298.11 | 12,288 | 2.395401 | 0.860231 |
| Candidate | 291.31 | 12,183 | 2.393355 | 0.861605 |

Candidate checkpoint `/data/chess/chess-engine-5/models/architecture-ablations.sdLWM7/wider-transformer-run-1`. Logs, configs, and the experiment code diff from `09c29cf` are in `/data/chess/chess-engine-5/models/architecture-ablations.sdLWM7`.

## Screening

Width 96 gave the lowest held-out policy loss of the five candidates. The gain of 0.002046 fell below the 0.005 threshold, and WDL loss increased.

The held-out screen required a policy loss improvement of at least 0.005, WDL loss within 0.005 of the control or better, and an improvement in combined loss.
