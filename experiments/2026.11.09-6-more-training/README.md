# More training

Test whether extending the deployed architecture and data recipe to five minutes improves playing strength.

Reused the five-minute control at 12,288 updates and 298.11 timed seconds. The deployed checkpoint used 5,904 updates. The control improved held-out policy loss from 2.413879 to 2.395401 and WDL loss from 0.869166 to 0.860231.

Gainer SPRT against `/data/chess/chess-engine-5/models/datagen-5000-experiment.SUaj4m/checkpoint`, with a cap of 256 color-swapped pairs. The inference benchmark assigned 800 visits per move to the baseline and 801 to the candidate.

- W/D/L 140/69/89
- Observed Elo +60.1
- SPRT verdict Passed

Promoted the five-minute control.

Logs and config snapshots are in `/data/chess/chess-engine-5/models/training-recipe-experiments.VioD2w`. Source revision `946aec0` plus the local schedule change.
