# Piece-count WDL MLP

Test whether a 16-unit hidden layer improves a value model that uses the same piece counts.

Trained on 250,301 positions for 30,151 optimizer steps (15.41 epochs).
Budget spent 1,780.53s (465.51s timed runtime + 0.3 × 4,383.40s estimated generation).
Held-out policy loss 2.2945 and WDL loss 0.8192.

## Gainer SPRT

Compared with `/data/chess/chess-engine-5/models/overnight-2026.11.10.Ky0bA8/02-deeper-labels-1200/candidate-1` at 789 candidate visits and 800 baseline visits per move. The match used a cap of 1,000 games and seed 48.

- W/D/L 125/88/83 over 296 games
- Observed Elo +49.6
- SPRT verdict Passed

Promoted the candidate.

Run artifacts in `/data/chess/chess-engine-5/models/overnight-2026.11.10.Ky0bA8/03-piece-count-wdl-mlp`.
