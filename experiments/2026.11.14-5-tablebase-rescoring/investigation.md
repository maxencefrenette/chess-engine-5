# Rescoring regression investigation

The regression was not caused by slower training: the candidate completed the same 37,260 optimizer steps in 310.36s versus 313.91s for the baseline. Rescoring increased setup from 1.33s to 5.35s, outside the timed budget.

With `value_mix = 1` and `policy_mix = 0`, the value head learned the tablebase labels, but became worse at predicting recorded outcomes and recognizing drawn endgames.

| Held-out metric | Baseline | Rescored |
|---|---:|---:|
| Tablebase WDL loss, 5,860 eligible positions | 0.9825 | 0.1895 |
| Game-result WDL loss, same positions | 0.4410 | 1.3537 |
| WDL loss on 522 tablebase draws from drawn games | 0.1837 | 0.5491 |
| Game-result WDL loss, positions with more than five pieces | 0.7451 | 0.7497 |
| Policy probability on WDL-preserving moves in winning endgames | 96.29% | 96.23% |

Of 2,615 changed training labels, 2,565 were recorded draws relabeled as tablebase wins or losses. These targets describe what perfect play can achieve, which the current policy and search may not convert.

## Conversion replay

Replay 28 tablebase-winning positions from distinct held-out games that originally ended in draws. Use 800 visits, four leaves per game, BF16 inference, and temperature zero. The defender randomly chooses among WDL-optimal moves. Starting positions have a halfmove clock of at most 20 and no repetition since the last zeroing move.

| Policy | Value | Wins | Draws | Draws reaching the 50-move limit |
|---|---|---:|---:|---:|
| Baseline | Baseline | 8 | 20 | 17 |
| Rescored | Rescored | 5 | 23 | 22 |
| Baseline | Rescored | 4 | 24 | 23 |
| Rescored | Baseline | 7 | 21 | 15 |

Swapping value predictions reproduces most of the conversion difference in this small diagnostic sample. The pattern is consistent with optimistic winning values that do not sufficiently distinguish progress from stalling. It does not establish how much of the full match regression this explains.

## Correctness audit

All 5,860 eligible held-out positions passed a board-only consistency check: root WDL equaled the best legal child result. This found no orientation or move/probe mismatch.

Reconstructing training-game histories exposed two losses incorrectly assigned to positions with an immediate repetition draw available. The probes use the halfmove clock but lack full repetition history. Among changed labels, 947 had a repetition since the last zeroing move; that does **not** mean all 947 labels are wrong. The two demonstrated errors alone do not explain the regression.

Before another rescoring experiment, fix the history handling. Then test `value_mix` between 0 and 1, to retain a signal about practical conversion.

Diagnostic code, predictions, audits, and replay trajectories: `/data/chess/chess-engine-5/models/tablebase-rescoring-2026.11.14/investigation`.
