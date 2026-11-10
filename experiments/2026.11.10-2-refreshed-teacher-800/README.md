# Refreshed teacher at 800 visits

Test whether self-play from the current baseline improves strength at the same search depth.

Generated 645,821 positions from 3,500 games in 2.10 hours at 800 visits and temperature 1.0. Teacher `/data/chess/chess-engine-5/models/thirty-minute-baseline.XHiM0X/run-1`.

Trained on 375,000 positions for 30,524 optimizer steps (10.42 epochs).
Budget spent 1,783.69s (468.67s timed runtime + 0.3 × 4,383.41s estimated generation).
Held-out policy loss 2.2799 and WDL loss 0.8630.

## Gainer SPRT

Compared with `/data/chess/chess-engine-5/models/thirty-minute-baseline.XHiM0X/run-1` at 798 candidate visits and 800 baseline visits per move. The match used a cap of 1,000 games and seed 46.

- W/D/L 388/246/366 over 1,000 games
- Observed Elo +7.6
- SPRT verdict Inconclusive

Run artifacts in `/data/chess/chess-engine-5/models/overnight-2026.11.10.Ky0bA8/01-refreshed-teacher-800`.
