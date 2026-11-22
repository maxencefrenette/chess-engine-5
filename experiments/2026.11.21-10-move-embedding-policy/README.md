# Attention-pooled move embeddings

Test whether pooling the board with one learned query, then scoring moves against a 64-dimensional MLP output, improves the policy over the baseline’s from/to square scoring.

| Policy head | Held-out policy loss | Held-out WDL loss | Total loss | Timed training (s) | Charged generation (s) | Budget (s) |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 1.49447 | 0.66215 | 2.15662 | 675.66 | 2920.28 | 3595.94 |
| Move embeddings | 1.93944 | 0.67440 | 2.61385 | 676.47 | 2920.28 | 3596.75 |

Policy loss accounted for 0.445 of the 0.457 increase in held-out total loss. We stopped the gainer before an SPRT verdict and kept the baseline.
