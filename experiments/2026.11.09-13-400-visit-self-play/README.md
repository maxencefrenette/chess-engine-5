# 400-visit self-play

Test whether cheaper 400-visit self-play labels improve playing strength when the savings buy more unique positions or optimizer updates.

Generated 887,682 positions from 5,000 games in 1.80 hours, using the same generator and temperature as the 800-visit data. Each recipe was charged only for positions it selected for training.

| Training positions | Steps | Epochs | Timed training (s) | Estimated generation (s) | Weighted total (s) | Policy loss | WDL loss |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 375,000 | 57,472 | 19.62 | 888.03 | 2737.59 | 1161.79 | 2.318391 | 0.834508 |
| 550,000 | 50,294 | 11.70 | 767.00 | 4015.19 | 1168.52 | 2.314296 | 0.837801 |
| 725,000 | 43,750 | 7.72 | 663.50 | 5292.70 | 1192.77 | 2.315564 | 0.838267 |

Selected the 550,000-position recipe by combined held-out loss on the common 400-visit split. The match compared it with the promoted 800-visit recipe.

## Match

- W/D/L 44/32/84
- Observed Elo -88.7
- SPRT verdict Failed

Dataset `/data/chess/chess-engine-5/training-data/v2/1794283385195233134-935635`. Logs and checkpoints are in `/data/chess/chess-engine-5/models/400-visit-sweep.MQj3lt`.
