# Positions versus epochs

Rebalance the 90-minute budget between unique training positions and repeated epochs.

| Positions | Epochs | Policy loss | WDL loss | Total loss | Training runtime (s) | Generation estimate (s) | Budget (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 291,816 (baseline) | 15.41 | 2.2043 | 0.7724 | 2.9767 | 293.72 | 4999.57 | 5293.29 |
| 300,000 | 13.24 | 2.2029 | 0.7804 | 2.9833 | 255.36 | 5139.77 | 5395.14 |
| 275,000 | 38.94 | 2.2042 | 0.8022 | 3.0065 | 711.57 | 4711.42 | 5422.99 |
| 250,000 | 67.18 | 2.2186 | 0.7723 | 2.9908 | 1122.44 | 4283.10 | 5405.54 |
| 200,000 | 147.62 | 2.2304 | 0.7729 | 3.0032 | 1965.64 | 3426.47 | 5392.10 |

Selected 300,000 positions for the lowest total held-out loss among the new candidates. None beat the baseline on total loss.

Gainer SPRT [0, 20] normalized Elo **failed** after 197 game pairs, at 800 visits per move for both models. Matched-speed check passed.

- W/D/L **104/156/134**
- Elo **−24.0 (−50.2, +2.1)**, 95% CI

Kept the current baseline and training recipe.

Artifacts: `/data/chess/chess-engine-5/models/budget-rebalance-2026.11.12`.
