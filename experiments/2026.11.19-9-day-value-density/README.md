# Value sampling and loss weight

Cross value sampling rates 0.5, 0.75, and 1 with loss weights 0.3 and 0.6. The incumbent is weight 0.3 and sample rate 0.5. Select the lowest total held-out loss within each weight.

| Candidate | Positions | Policy loss | Mixed WDL loss | Total loss | Training (s) | Generation (s) | Budget (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| fresh | 309,700 | 1.80908 | 0.68782 | 2.49690 | 356.21 | 3245.91 | 3602.12 |
| weight-0.3-rate-0.75 | 309,700 | 1.80635 | 0.68573 | 2.49209 | 356.49 | 3245.91 | 3602.40 |
| weight-0.3-rate-1.0 | 309,700 | 1.80524 | 0.68754 | 2.49279 | 353.78 | 3245.91 | 3599.69 |
| weight-0.6-rate-0.5 | 309,700 | 1.80258 | 0.69028 | 2.49286 | 355.54 | 3245.91 | 3601.45 |
| weight-0.6-rate-0.75 | 309,700 | 1.80295 | 0.68825 | 2.49120 | 357.13 | 3245.91 | 3603.04 |
| weight-0.6-rate-1.0 | 309,700 | 1.80310 | 0.68914 | 2.49225 | 357.43 | 3245.91 | 3603.34 |

| Candidate | Opponent | Visits (candidate/baseline) | W/D/L | nElo (95% CI) | Gainer SPRT |
|---|---|---:|---|---|---|
| weight-0.3-rate-0.75 | fresh | 800/800 | 408/494/420 | -0.8 (-20.8, +19.9) | failed |
| weight-0.6-rate-0.75 | fresh | 800/800 | 490/503/417 | +19.7 (-0.4, +38.8) | passed |

Promoted weight-0.6-rate-0.75.
