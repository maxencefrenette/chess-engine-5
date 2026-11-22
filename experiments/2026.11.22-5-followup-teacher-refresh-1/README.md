# Fresh self-play with softer search priors

Test whether one generation of self-play from the incumbent improves its successor at 1,000 visits and 18 epochs. Generated 1,397,050 positions from 10,000 games using `epochs-with-reused-search-data/epochs-18`.

The incumbent trained on 994,000 positions generated at prior temperature 1.35. This generation used the incumbent as its teacher and prior temperature 2.00. Both players used 2.00 in the match.

| Candidate | Unique positions | Epochs | Training (s) | Charged generation (s) | Budget (s) | Policy loss | WDL loss | Total loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fresh data | 876,800 | 18 | 890.3 | 2705.5 | 3595.8 | 1.9830 | 0.5410 | 2.5239 |

The [0, 15] nElo gainer for **fresh data** on seed 201 **failed** after 202 pairs. Candidate W/D/L 140/68/196, nElo -50.7 (-85.5, -15.8), 95% CI.

Kept the incumbent.
