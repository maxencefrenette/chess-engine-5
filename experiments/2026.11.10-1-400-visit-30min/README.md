# 400-visit self-play

Test whether cheaper 400-visit self-play labels improve playing strength when the savings buy more unique positions or optimizer updates.

Reused the existing 400-visit dataset. Each run targeted 1,800 seconds including 0.3 times the estimated generation time for its training positions.

| Training positions | Steps | Epochs | Timed training (s) | Estimated generation (s) | Weighted total (s) | Policy loss | WDL loss |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 375,000 | 63,903 | 21.81 | 955.16 | 2737.59 | 1776.44 | 2.315581 | 0.833024 |
| 475,000 | 49,589 | 13.36 | 761.64 | 3467.59 | 1801.92 | 2.316546 | 0.834788 |
| 575,000 | 35,275 | 7.85 | 541.76 | 4197.70 | 1801.07 | 2.318855 | 0.835772 |
| 675,000 | 20,961 | 3.97 | 310.38 | 4927.69 | 1788.69 | 2.323062 | 0.836470 |

Selected the 375,000-position recipe by combined held-out loss on the common 400-visit split. The match compared it with the current 800-visit recipe.

## Match

- W/D/L 27/29/78
- Observed Elo -139.2
- SPRT verdict Failed

Dataset `/data/chess/chess-engine-5/training-data/v2/1794283385195233134-935635`. Logs and checkpoints are in `/data/chess/chess-engine-5/models/400-visit-30min-sweep.8uJIjn`.
