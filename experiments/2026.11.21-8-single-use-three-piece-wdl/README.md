# Single-use three-piece WDL samples

Test whether 12,000 unique tablebase positions improve playing strength when each supplies one WDL target during the whole run. Pawn, rook and queen versus bare king each contributed 4,000 legal nonterminal positions. Training mixed 800 new tablebase positions into each of 15 self-play epochs.

| Model | Self-play positions | Policy loss | WDL loss | Total held-out loss | Timed training (s) | Charged self-play (s) | Budget (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 779,600 | 1.49447 | 0.66215 | 2.15662 | 675.66 | 2920.28 | 3595.94 |
| Three-piece | 779,600 | 1.50551 | 0.66515 | 2.17065 | 669.52 | 2920.28 | 3589.80 |

WDL accuracy on 4,096 held-out positions per family, at a fresh halfmove clock with no castling rights. The test excludes self-play and sampled training geometries, including reflections and color reversals. Each cell gives baseline → candidate accuracy for the side to move.

| Extra piece | Win | Draw | Loss | Overall |
|---|---:|---:|---:|---:|
| Pawn | 79.7% → 90.4% | 90.6% → 61.0% | 34.8% → 87.7% | 69.9% → 79.7% |
| Rook | 100.0% → 100.0% | 100.0% → 96.5% | 95.4% → 99.4% | 97.8% → 99.5% |
| Queen | 100.0% → 100.0% | 38.5% → 52.9% | 100.0% → 100.0% | 96.3% → 97.2% |

**Failed** the matched-speed gainer after 176 pairs at 800 visits per move. W/D/L **104/108/140**, nElo **-38.6 (-75.9, -1.0)** (95% CI). Kept the baseline.
