# Repeated self-play refreshes

Test how many successive generations of fresh 800-visit self-play improve the 60-minute baseline. Each candidate trained for 15 epochs and played its teacher at 800 visits per move.

| Candidate | Unique positions | Epochs | Policy loss | WDL loss | Total loss | Training (s) | Charged generation (s) | Budget (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Round 1 | 776,800 | 15 | 1.58854 | 0.64769 | 2.23624 | 672.19 | 2923.05 | 3595.25 |
| Round 2 | 776,100 | 15 | 1.54130 | 0.63636 | 2.17766 | 676.68 | 2928.35 | 3605.03 |
| Round 3 | 779,600 | 15 | 1.49447 | 0.66215 | 2.15662 | 675.66 | 2920.28 | 3595.94 |
| Round 4 | 783,100 | 15 | 1.48506 | 0.66287 | 2.14793 | 676.21 | 2921.17 | 3597.37 |

| Candidate | Opponent | Pairs | W/D/L | nElo (95% CI) | Gainer |
|---|---|---:|---|---|---|
| Round 1 | Starting baseline | 500 | 435/287/278 | +65.3 (+43.7, +86.9) | passed |
| Round 2 | Round 1 | 529 | 389/349/320 | +24.3 (+1.5, +46.4) | passed |
| Round 3 | Round 2 | 500 | 363/344/293 | +30.2 (+8.6, +51.7) | passed |
| Round 4 | Round 3 | 407 | 257/276/281 | -9.4 (-34.4, +16.1) | failed |

Promoted rounds 1 through 3. Stopped after round 4.
