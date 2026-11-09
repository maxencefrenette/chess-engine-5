# One layer inference

Test whether a one-layer policy model can use faster inference to gain playing strength at matched inference time.

Across three paired benchmarks at 256 positions, the median was 1.572 ms for one layer and 2.471 ms for two layers, a 1.57× throughput ratio. The predeclared screen required at least 1.20×. The existing one-layer checkpoint had policy loss 2.394013 and WDL loss 0.857868.

Logs and config snapshots are in `/data/chess/chess-engine-5/models/training-recipe-experiments.VioD2w`. Source revision `946aec0` plus the local schedule change.

## Baseline match

Gainer SPRT against `/data/chess/chess-engine-5/models/training-recipe-experiments.VioD2w/five-minute-control`, with a cap of 256 color-swapped pairs. The inference benchmark assigned 800 visits per move to the baseline and 1,251 to the candidate.

- W/D/L 188/95/139
- Observed Elo +40.5
- SPRT verdict Passed

Promoted the candidate.
