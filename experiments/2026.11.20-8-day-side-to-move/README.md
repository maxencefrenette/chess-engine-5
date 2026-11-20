# Side-to-move board encoding

Test whether flipping ranks and exchanging piece colors on Black’s turn improves playing strength. Map legal moves and reply targets into the same coordinate frame.

| Candidate | Positions | Policy loss | Mixed WDL loss | Total loss | Training (s) | Generation (s) | Budget (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Absolute | 383,300 | 1.68476 | 0.69077 | 2.37553 | 331.08 | 3268.51 | 3599.58 |
| Side to move | 383,300 | 1.73799 | 0.68796 | 2.42595 | 330.97 | 3268.51 | 3599.48 |

The parity benchmark measured a 0.29% slowdown, so the match used time-matched visits.

| Candidate | Opponent | Visits (candidate/baseline) | W/D/L | nElo (95% CI) | Gainer SPRT |
|---|---|---:|---|---|---|
| Side to move | Absolute | 798/800 | 516/239/245 | +111.9 (+90.3, +133.4) | passed |

The gainer passed at pair 79. Promoted the side-to-move checkpoint after 500 pairs.
