# 1,200-visit self-play

Test whether deeper self-play labels from the same teacher justify fewer training positions.

Generated 357,955 positions from 2,000 games in 1.74 hours at 1,200 visits and temperature 1.0. Teacher `/data/chess/chess-engine-5/models/thirty-minute-baseline.XHiM0X/run-1`.

Kept the teacher fixed and targeted the same generation charge as the refreshed 800-visit recipe.

Trained on 250,301 positions for 31,837 optimizer steps (16.28 epochs).
Budget spent 1,793.01s (477.99s timed runtime + 0.3 × 4,383.40s estimated generation).
Held-out policy loss 2.2930 and WDL loss 0.8581.

## Gainer SPRT

Compared with `/data/chess/chess-engine-5/models/thirty-minute-baseline.XHiM0X/run-1` at 799 candidate visits and 800 baseline visits per move. The match used a cap of 1,000 games and seed 47.

- W/D/L 96/53/51 over 200 games
- Observed Elo +79.5
- SPRT verdict Passed

Promoted the candidate.

Run artifacts in `/data/chess/chess-engine-5/models/overnight-2026.11.10.Ky0bA8/02-deeper-labels-1200`.
