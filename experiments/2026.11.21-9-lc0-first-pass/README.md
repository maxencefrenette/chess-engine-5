# Lc0 first-pass tablebase rescoring

Test whether carrying five-piece tablebase outcomes back through self-play games improves playing strength. After a capture or pawn move resets the fifty-move clock, a successful WDL probe supplies the outcome back to the previous successful probe, or to the start of the game. The value target mixes this outcome equally with root WDL. [Lc0 first pass](https://github.com/LeelaChessZero/lc0/blob/bf15d558e178aa6415e40ab476bd1a3617828f1f/src/trainingdata/rescorer.cc#L560-L608)

The 8,900 probes covered 195,806 of 585,549 selected value targets. Of these, 17,427 differed from the played result, including 16,142 above five pieces. The first probe disagreed with the game result in 171 of the 1,800 training games that reached a probe.

| Model | Held-out policy loss | Held-out WDL loss | Total loss | Timed training (s) | Charged generation (s) | Budget (s) |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 1.49447 | 0.66215 | 2.15662 | 675.66 | 2920.28 | 3595.94 |
| First-pass rescoring | 1.49399 | 0.66067 | 2.15465 | 675.64 | 2923.86 | 3599.50 |

**Failed** the [0, 20] matched-speed gainer after 3,735 pairs at 800 visits per move. Candidate W/D/L **2,547/2,507/2,416**. Estimated nElo **+9.7 (−2.5, +22.4)** (SPRT-adjusted approximate 95% CI).

After changing the gainer bounds to [0, 15] nElo, we resumed the same match for another 1,343 pairs. The [0, 15] gainer **failed** at 5,078 cumulative pairs. Candidate W/D/L **3,427/3,423/3,306**. Estimated nElo **+7.0 (−2.1, +16.8)** (SPRT-adjusted approximate 95% CI). Restored the previous baseline and disabled rescoring for training.
