# Subtree reuse in self-play

Test whether carrying search forward after each self-play move buys enough extra training positions to improve strength within 60 weighted minutes. The played child becomes the next root. Its outgoing visits count toward 800, and its first network evaluation is excluded from the new root’s Q and draw averages.

| 500-game benchmark | Positions | Generation (s) | Positions/s | Fresh network evaluations/position |
|---|---:|---:|---:|---:|
| Fresh search | 65,579 | 236.45 | 277.4 | 768.9 |
| Subtree reuse | 68,220 | 128.46 | 531.1 | 376.0 |

Both benchmarks used the incumbent checkpoint and seed 43. Production used seed 42.

Production generated 1,626,518 positions from 12,000 games at 512.3 positions/s. All 1,626,518 policies summed to 800 visits. After holding out 1,200 whole games, we trained from scratch on 1,277,100 positions for 15 epochs, 63.8% more unique positions than the incumbent.

| Training | Unique positions | Epochs | Timed training (s) | Charged generation (s) | Budget (s) |
|---|---:|---:|---:|---:|---:|
| Incumbent | 779,600 | 15 | 675.66 | 2920.28 | 3595.94 |
| Subtree reuse | 1,277,100 | 15 | 1086.33 | 2493.15 | 3579.48 |

The matched-speed [0, 15] nElo gainer **passed** after 796 game pairs at 800 visits per side. Candidate W/D/L **570/534/488**, estimated **+20.2 nElo (+2.0, +37.9)**, 95% CI. Three preliminary speed checks rejected differences of 0.12–0.21%. Eight paired benchmarks with reversed model order gave a time ratio of 0.9999 (95% CI 0.9993–1.0006), and the official match passed its speed check.

Promoted the candidate and enabled subtree reuse for self-play.
