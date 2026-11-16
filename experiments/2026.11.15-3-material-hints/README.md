# Material WDL hints

Test whether material-bucket WDL hints with learned hint and trust embeddings improve held-out loss and playing strength.

Each game contributes at most one unit per exact material bucket. Training hints include the current game; held-out games never contribute. Trust depends on game count and WDL concentration.

## Hint design

| Injection | Minimum games | Hint dropout | Policy loss | WDL loss | Total loss |
|---|---:|---:|---:|---:|---:|
| No hints | — | — | 1.959161 | 0.705553 | 2.664713 |
| Early | 20 | 0% | 1.961279 | 0.714564 | 2.675843 |
| Early | 100 | 0% | 1.961439 | 0.705910 | 2.667349 |
| Early | 5 | 0% | 1.963242 | 0.714794 | 2.678036 |
| Early + value head | 20 | 0% | 1.961256 | 0.704635 | 2.665891 |
| Early + value head | 5 | 0% | 1.963330 | 0.717069 | 2.680399 |
| Early + value head | 20 | 50% | 1.958644 | 0.702071 | 2.660716 |

Selected 20-game support, early + value-head injection, and 50% training hint dropout. Total held-out loss improved by 0.003998.

Timed runtime: 332.68–344.40s. Generation charge: 3286.51s. Weighted budget: 3619.19–3630.91s. The selected candidate used 3627.43s.

## Training tuning

Cosine schedules use 5% warmup and decay to 10% of the peak learning rate.

| Hints | Schedule | Learning rate | Hint dropout | Policy loss | WDL loss | Total loss |
|---|---|---:|---:|---:|---:|---:|
| Initial selected recipe | Constant | 0.001 | 50% | 1.958644 | 0.702071 | 2.660716 |
| Enabled | Constant | 0.0005 | 50% | 1.960174 | 0.705699 | 2.665873 |
| Enabled | Constant | 0.00075 | 50% | 1.959809 | 0.704033 | 2.663842 |
| Enabled | Constant | 0.0015 | 50% | 1.965512 | 0.707607 | 2.673119 |
| Enabled | Cosine | 0.001 | 50% | 1.956491 | 0.683537 | 2.640029 |
| Enabled | Cosine | 0.002 | 50% | 1.939457 | 0.687458 | 2.626915 |
| Enabled | Cosine | 0.002 | 25% | 1.939222 | 0.688630 | 2.627852 |
| Enabled | Cosine | 0.002 | 75% | 1.939953 | 0.686767 | 2.626720 |
| Disabled | Cosine | 0.002 | — | 1.938736 | 0.690366 | 2.629102 |

Selected cosine decay with peak learning rate 0.002 and 75% hint dropout. Total loss improved by 0.033995 over the initial selected recipe and 0.037993 over baseline. The no-hint control shows that most of the gain comes from the schedule; hints improve total loss by another 0.002382.

Timed runtime: 309.04–343.33s. Generation charge: 3286.51s. Weighted budget: 3595.55–3629.84s. The selected candidate used 3629.84s.

## Matches

Both candidates faced the same baseline with gainer SPRT bounds [0, 20] nElo and time-matched visits.

| Candidate | Pairs | W/D/L | nElo (95% CI) | Verdict | Visits (candidate / baseline) | Inference ms (candidate / baseline, batch 512) |
|---|---:|---|---|---|---|---|
| Initial: constant LR 0.001, 50% dropout | 203 | 121/134/151 | −30.8 (−65.2, +4.0) | Failed | 742 / 800 | 2.335 / 2.167 |
| Tuned: cosine LR 0.002, 75% dropout | 1,634 | 1041/1298/929 | +12.5 (−2.3, +25.9) | Passed | 744 / 800 | 2.328 / 2.166 |

Promoted the tuned candidate and carried forward its training recipe.

Artifacts:
- Hint design: `/data/chess/chess-engine-5/models/material-hints-2026.11.15`
- Training tuning: `/data/chess/chess-engine-5/models/material-hint-tuning-2026.11.15`
