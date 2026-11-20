# Attention pooling for value

Compare mean and single-query attention pooling at hidden widths 16 and 64. The incumbent uses mean pooling at width 16. The query starts at zero, giving equal weight to every square.

| Candidate | Positions | Policy loss | Mixed WDL loss | Total loss | Training (s) | Generation (s) | Budget (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| weight-0.6-rate-0.75 | 309,700 | 1.80295 | 0.68825 | 2.49120 | 357.13 | 3245.91 | 3603.04 |
| pooled-64 | 309,700 | 1.80376 | 0.68992 | 2.49369 | 355.23 | 3245.91 | 3601.14 |
| attention-16 | 309,700 | 1.80686 | 0.68650 | 2.49336 | 371.34 | 3245.91 | 3617.25 |
| attention-64 | 309,700 | 1.80888 | 0.68694 | 2.49582 | 370.54 | 3245.91 | 3616.45 |

| Candidate | Opponent | Visits (candidate/baseline) | W/D/L | nElo (95% CI) | Gainer SPRT |
|---|---|---:|---|---|---|
| attention-16 | weight-0.6-rate-0.75 | 775/800 | 378/327/313 | +24.5 (+1.6, +46.7) | passed |

Promoted attention-16.
