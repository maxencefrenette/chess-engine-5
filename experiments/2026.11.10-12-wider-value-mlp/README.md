# Wider piece-count value MLP

Test whether more hidden units improve the piece-count WDL model.

| Recipe | Epochs | Policy loss | WDL loss |
|---|---:|---:|---:|
| Baseline | 15.41 | 2.247751 | 0.820616 |
| hidden-32 | 15.41 | 2.247751 | 0.820989 |
| hidden-64 | 15.41 | 2.247751 | 0.820598 |

Selected hidden-64 by held-out WDL loss.

## Gainer SPRT

- W/D/L 30/75/49 over 154 games
- Observed Elo -43.1
- SPRT verdict Failed
- Visits per move 765 candidate, 800 baseline

Run artifacts in `/data/chess/chess-engine-5/models/value-capacity-2026.11.10.Hzhknn/wider-value-mlp`.
