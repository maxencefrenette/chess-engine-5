# Predict the opponent’s reply

Train an auxiliary policy from the parent position conditioned on the played move. Of 318,400 paid positions, 218,540 had an adjacent selected position in the same game to supply a reply target. Discard the auxiliary head before checkpointing.

Baseline held-out total loss: 2.535242.

| Setting | Policy | WDL | Total loss | Runtime (s) | Generation (s) | Total budget (s) |
|---|---:|---:|---:|---:|---:|---:|
| reply-0.05 | 1.896316 | 0.648269 | 2.544584 | 381.11 | 3277.43 | 3658.54 |
| reply-0.15 | 1.869305 | 0.647311 | 2.516616 | 383.56 | 3277.43 | 3660.99 |

Selected reply-0.15. Matched-speed gainer passed after 658 game pairs. W/D/L 449/487/380. Observed nElo (95% CI) +20.6 (-0.0, +40.4).

Promoted the candidate.

Artifacts: `/data/chess/chess-engine-5/models/supervision-queue-2026.11.18/opponent-reply-policy`.
