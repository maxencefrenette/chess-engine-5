# Self-play at 600 versus 800 visits

Test whether more unique positions compensate for shallower self-play searches. The incumbent supplies the 800-visit control. Its teacher generated the 600-visit dataset. Both models trained for 15 epochs and played the match at 800 visits per move. Each arm’s held-out losses use its own search targets.

| Candidate | Unique positions | Epochs | Policy loss | WDL loss | Total loss | Training (s) | Charged generation (s) | Budget (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 800 visits | 779,600 | 15 | 1.49447 | 0.66215 | 2.15662 | 675.66 | 2920.28 | 3595.94 |
| 600 visits | 969,500 | 15 | 1.48242 | 0.65859 | 2.14101 | 838.51 | 2759.39 | 3597.90 |

| Candidate | Opponent | Pairs | W/D/L | nElo (95% CI) | Gainer |
|---|---|---:|---|---|---|
| 600 visits | 800 visits | 222 | 139/134/171 | -27.0 (-60.0, +6.3) | failed |

Kept 800 visits.
