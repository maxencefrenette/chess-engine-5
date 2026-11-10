# One layer inference

Test whether a one-layer policy model can use faster inference to gain playing strength at matched inference time.

Across three paired benchmarks at 256 positions, the median was 1.572 ms for one layer and 2.471 ms for two layers, a 1.57× throughput ratio. The predeclared screen required at least 1.20×. The existing one-layer checkpoint had policy loss 2.394013 and WDL loss 0.857868.

Logs and config snapshots are in `/data/chess/chess-engine-5/models/training-recipe-experiments.VioD2w`. Source revision `946aec0` plus the local schedule change.

## Match

The inference benchmark assigned 1,251 visits per move to the one-layer model and 800 to the two-layer model.

- W/D/L 188/95/139
- Observed Elo +40.5
- SPRT verdict Passed

Promoted the candidate.
