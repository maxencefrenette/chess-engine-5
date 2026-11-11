# Epochs versus unique training positions

Test how to split the 30-minute weighted budget between training on more unique positions and taking more passes.

| Recipe | Positions | Epochs | Timed runtime (s) | Estimated generation (s) | Weighted total (s) | Policy loss | WDL loss | Total loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 250,301 | 15.41 | 469.78 | 4333.66 | 1769.88 | 2.2478 | 0.8206 | 3.0684 |
| positions-200241 | 200,241 | 30.62 | 747.53 | 3466.94 | 1787.62 | 2.2619 | 0.8191 | 3.0810 |
| positions-275331 | 275,331 | 11.02 | 368.20 | 4766.96 | 1798.29 | 2.2520 | 0.8177 | 3.0697 |

No candidate improved baseline held-out combined policy and WDL loss.

Run artifacts in `/data/chess/chess-engine-5/models/value-capacity-2026.11.10.Hzhknn/epochs-vs-positions`.
