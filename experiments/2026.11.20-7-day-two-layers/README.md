# Two-layer transformers

Sweep width and learning rate at two layers. Compare the finalist with the one-layer incumbent using a time-matched non-regression SPRT [−20, 0] nElo.

| Layers | Width | Learning rate | Positions | Policy loss | WDL loss | Total loss | Batch-512 inference (ms) | Visits | Training (s) | Generation (s) | Budget (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.002 | 383,300 | 1.68476 | 0.69077 | 2.37553 | 2.247 | 800 | 331.08 | 3268.51 | 3599.58 |
| 2 | 32 | 0.001 | 366,700 | 1.80411 | 0.69411 | 2.49823 | 2.543 | 704 | 472.42 | 3126.96 | 3599.38 |
| 2 | 32 | 0.002 | 366,700 | 1.71893 | 0.69254 | 2.41147 | 2.552 | 704 | 473.10 | 3126.96 | 3600.06 |
| 2 | 48 | 0.001 | 366,400 | 1.73692 | 0.68832 | 2.42524 | 3.177 | 565 | 470.24 | 3124.41 | 3594.64 |
| 2 | 48 | 0.002 | 366,400 | 1.69473 | 0.68833 | 2.38307 | 3.184 | 566 | 472.70 | 3124.41 | 3597.11 |
| 2 | 64 | 0.001 | 367,000 | 1.71131 | 0.69405 | 2.40536 | 3.686 | 488 | 465.61 | 3129.52 | 3595.13 |
| 2 | 64 | 0.002 | 367,000 | 1.68891 | 0.69524 | 2.38416 | 3.685 | 488 | 465.36 | 3129.52 | 3594.88 |

The two-layer widths 48 and 64 finished within 0.005 total loss of the best candidate. Width 48 earned more visits and also had the lowest loss.

| Width | Learning rate | Match visits (candidate/baseline) | W/D/L | nElo (95% CI) | Non-regression SPRT |
|---:|---:|---:|---|---|---|
| 48 | 0.002 | 565/800 | 46/54/96 | -111.6 (-164.3, -58.8) | failed |

Kept one layer.
