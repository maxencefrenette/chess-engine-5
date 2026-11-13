# Larger transformer

Spend training compute on a larger transformer instead of more epochs within the 90-minute budget.

| Model | Epochs | Policy loss | WDL loss | Total loss | Timed runtime (s) | Generation estimate (s) | Budget (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline: 1 layer, width 64, FFN 128 | 15.41 | 2.2043 | 0.7724 | 2.9767 | 293.72 | 4999.57 | 5293.29 |
| 2 layers, width 64, FFN 128 | 9.62 | 2.2071 | 0.7938 | 3.0008 | 293.23 | 4999.57 | 5292.80 |
| 1 layer, width 128, FFN 256 | 15.38 | 2.2038 | 0.7774 | 2.9813 | 287.76 | 4999.57 | 5287.33 |

Selected width 128 for the lowest total held-out loss among the larger models. Neither beat the baseline on total loss.

The wider model trained at nearly the baseline's speed, but inference took 3.915 ms versus 2.685 ms per batch. Its time-matched gainer ran at 549 visits per move versus 800 for baseline.

Gainer SPRT [0, 20] normalized Elo **failed** after 63 game pairs.

- W/D/L **21/46/59**
- Elo **−105.6 (−157.2, −58.1)**, 95% CI

Kept the current baseline and training recipe.

Artifacts: `/data/chess/chess-engine-5/models/larger-model-2026.11.13`.
