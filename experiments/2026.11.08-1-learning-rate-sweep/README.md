# Learning-rate sweep

Find whether changing the AdamW learning rate improves held-out policy and WDL loss and produces a stronger checkpoint.

| Learning rate | Policy loss | WDL loss | Combined loss |
|---:|---:|---:|---:|
| 0.00025 | 2.440579 | 0.862687 | 3.303265 |
| 0.0005 | 2.428038 | 0.837990 | 3.266028 |
| 0.001 (control) | 2.421967 | 0.822216 | 3.244183 |
| 0.002 | 2.415935 | 0.811877 | 3.227812 |
| 0.004 | 2.438327 | 0.807517 | 3.245845 |

The lowest combined loss was at `0.002`. Checkpoints and logs are in `/data/chess/chess-engine-5/models/lr-sweep-2026-11-08-msdv_w0t`.

## Match

- W/D/L: 19/31/68
- Observed Elo: -153.5
- SPRT verdict: Failed
