# Fresh self-play from the incumbent

Test whether one generation of self-play from the incumbent improves its successor at 800 visits and 15 epochs. Generated 1,798,852 positions from 13,000 games using `2026.11.21-13-subtree-reuse/candidate`.

Self-play also used the promoted search settings: first_play = { mode = "parent", reduction = 0.15 }, policy_temperature = 1.35. The match measures the combined generational change.

| Candidate | Unique positions | Epochs | Training (s) | Charged generation (s) | Budget (s) | Policy loss | WDL loss | Total loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fresh data | 1,215,100 | 15 | 1040.3 | 2567.0 | 3607.3 | 1.6328 | 0.6519 | 2.2847 |

The [0, 15] nElo gainer for **fresh data** on seed 131 **passed** after 500 pairs. Candidate W/D/L 454/257/289, nElo +67.1 (+45.5, +88.6), 95% CI.

Promoted the fresh-data checkpoint.
