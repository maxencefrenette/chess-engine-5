# Transformer WDL head

Test whether a WDL head trained jointly with policy on the shared transformer improves playing strength.

Trained on 250,301 positions for 29,393 optimizer steps (15.03 epochs).
Budget spent 1,790.60s (475.58s timed runtime + 0.3 × 4,383.40s estimated generation).
Held-out policy loss 2.3198 and WDL loss 2.4000.

Training WDL loss fell from 0.8340 in epoch 1 to 0.2227 in epoch 15, while final held-out WDL loss was 2.4000 versus the MLP baseline's 0.8192 on the same games. This points to substantial value overfitting.

## Gainer SPRT

Compared with `/data/chess/chess-engine-5/models/overnight-2026.11.10.Ky0bA8/03-piece-count-wdl-mlp/candidate-1` at 805 candidate visits and 800 baseline visits per move. The match used a cap of 1,000 games and seed 49.

- W/D/L 1/25/74 over 100 games
- Observed Elo -322.7
- SPRT verdict Failed

Run artifacts in `/data/chess/chess-engine-5/models/overnight-2026.11.10.Ky0bA8/04-transformer-wdl-head`.
