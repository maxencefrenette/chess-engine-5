# Two-stage attention policy head

Test whether a 128-wide policy head can improve move prediction by reading the squares twice. A learned linear map projects each 64-feature square to 128 features. The first attention stage uses a learned query. Its MLP supplies the query for the second stage, whose pooled output is added to the first MLP’s output before the final MLP scores moves against their embeddings.

| Policy head | Held-out policy loss | Held-out WDL loss | Total loss | Timed training (s) | Charged generation (s) | Budget (s) |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 1.49447 | 0.66215 | 2.15662 | 675.66 | 2920.28 | 3595.94 |
| One stage, width 64 | 1.93944 | 0.67440 | 2.61385 | 676.47 | 2920.28 | 3596.75 |
| Two stages, width 128 | 1.94596 | 0.67742 | 2.62338 | 767.65 | 2920.28 | 3687.93 |

The two-stage head increased held-out total loss by 0.00953 over the one-stage head and 0.46676 over the baseline. We skipped the gainer and kept the baseline.
