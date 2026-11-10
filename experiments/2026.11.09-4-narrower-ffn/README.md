# Narrower ffn

Test whether halving the transformer feed-forward width from 128 to 64 improves the model under the same wall-clock budget.

Training override `model.ffn_width=64`. Step counts were calibrated for the same training-time window.

| Run | Timed seconds | Optimizer steps | Held-out policy loss | Held-out WDL loss |
|---|---:|---:|---:|---:|
| Five-minute control | 298.11 | 12,288 | 2.395401 | 0.860231 |
| Candidate | 305.70 | 12,521 | 2.398436 | 0.860236 |

Candidate checkpoint `/data/chess/chess-engine-5/models/architecture-ablations.sdLWM7/narrower-ffn-run-1`. Logs, configs, and the experiment code diff from `09c29cf` are in `/data/chess/chess-engine-5/models/architecture-ablations.sdLWM7`.

## Screening

Halving feed-forward width gave little change in updates completed and increased held-out policy loss.

The held-out screen required a policy loss improvement of at least 0.005, WDL loss within 0.005 of the control or better, and an improvement in combined loss.
