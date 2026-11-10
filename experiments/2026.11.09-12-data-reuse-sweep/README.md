# Data reuse under a fixed training budget

Test whether training on fewer unique positions and making more passes improves playing strength under the 20-minute weighted budget.

| Training positions | Steps | Epochs | Timed training (s) | Estimated generation (s) | Weighted total (s) | Policy loss | WDL loss |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 755,637 (control) | 7,160 | 1.21 | 141.67 | 10932.38 | 1234.91 | 2.417474 | 0.867161 |
| 375,000 | 42,886 | 14.64 | 666.76 | 5425.49 | 1209.31 | 2.386696 | 0.854795 |
| 500,000 | 29,975 | 7.67 | 453.84 | 7233.95 | 1177.23 | 2.389800 | 0.856298 |

The screen required a policy loss gain of at least 0.005, a WDL loss increase of at most 0.005, and a lower combined loss.

Selected the 375,000-position checkpoint for a gainer match.

## Match

- W/D/L 162/90/114
- Observed Elo +45.8
- SPRT verdict Passed

Promoted the candidate.

Logs and checkpoints are in `/data/chess/chess-engine-5/models/data-reuse-sweep.5w16bD`.
