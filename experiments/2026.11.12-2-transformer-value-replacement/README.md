# Transformer WDL replacement

Replace the CPU piece-count value model with a transformer WDL head without losing playing strength.

Test random value masking, a smaller per-square head, mean pooling, and reduced value-loss weight.

The baseline held-out losses are policy **2.2042**, WDL **0.7576**, total **2.9617**.

| Head | Value sample rate | Value-loss weight | Policy loss | WDL loss | Total loss | Timed runtime (s) | Weighted budget (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Per-square (1 channel, hidden 16) | 0.125 | 1 | 2.2664 | 1.1812 | 3.4477 | 300.76 | 1800.63 |
| Per-square (1 channel, hidden 16) | 0.03125 | 1 | 2.2750 | 0.8783 | 3.1533 | 302.18 | 1802.05 |
| Per-square (1 channel, hidden 16) | 0.0078125 | 1 | 2.2509 | 1.0386 | 3.2894 | 291.54 | 1791.42 |
| Mean pool (hidden 16) | 0.03125 | 1 | 2.2637 | 0.8657 | 3.1294 | 292.74 | 1792.61 |
| Mean pool (hidden 16) | 0.03125 | 0.1 | 2.2047 | 0.7890 | 2.9937 | 294.25 | 1794.12 |
| Mean pool (hidden 16) | 0.03125 | 0.03 | 2.2032 | 0.7893 | 2.9925 | 292.86 | 1792.74 |
| Mean pool (hidden 16) | 0.125 | 0.03 | 2.2043 | 0.7724 | 2.9767 | 293.72 | 1793.60 |

Selected the mean-pooled head with 1/8 value sampling and value-loss weight 0.03. Policy loss is essentially unchanged, with a small WDL loss increase.

Non-regression SPRT [−20, 0] normalized Elo **passed** after 280 game pairs:

- W/D/L **176/236/148**
- Elo **+14.8 (−8.4, +37.8)**, 95% CI
- Visits per move: **810 candidate, 800 baseline**

Promoted the candidate for evaluation and self-play. Removed the piece-count value model and its feature plumbing.

Artifacts: `/data/chess/chess-engine-5/models/transformer-value-2026.11.12`.
