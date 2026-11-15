# Tablebase value rescoring

Test whether rescoring training endgames with ChessTB WDL and DTZ improves playing strength within the 60-minute budget.

Use `value_mix = 1` and `policy_mix = 0` to rescore selected value targets with five pieces or fewer, using the side to move and actual halfmove clock. Held-out targets remain the recorded game outcomes.

Probed **5,193** training value labels and changed **2,615**.

| Original target → rescored | Loss | Draw | Win |
|---|---:|---:|---:|
| Loss | 1042 | 21 | 4 |
| Draw | 1282 | 527 | 1283 |
| Win | 6 | 19 | 1009 |

Timed training: 310.68s; generation charge: 3286.51s; total budget: 3597.19s.

Policy loss: 1.958656; WDL loss: 0.828259; total loss: 2.786915.

Gainer SPRT [0, 20] nElo **failed** after **103 game pairs**, at 800 visits per move for both models.

- W/D/L **32/110/64**
- nElo **-79.2 (-128.7, -29.5)**, 95% CI

Kept the previous baseline.

[Follow-up investigation](investigation.md): the value predictions improved against tablebase targets but reduced endgame conversion in diagnostic replays. Training speed was not the cause.

Tables: `/data/chess/tablebases/chesstb-5` (458 WDL/DTZ files, 780,421,136 bytes).

Artifacts: `/data/chess/chess-engine-5/models/tablebase-rescoring-2026.11.14`.
