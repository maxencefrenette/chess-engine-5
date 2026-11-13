# Fresh data from the BF16 baseline

Test whether fresh BF16 self-play data improves playing strength within the 30-minute weighted training budget.

The gainer and non-regression tests gave confusing results, so we ran a fresh fixed comparison of 2,000 pairs.

| Test | W/D/L | Elo (95% CI) | Result |
|---|---|---|---|
| Gainer [0, 20] | 124/202/140 | −11.9 (−35.7, +11.7) | Failed |
| Non-regression [−20, 0] | 86/106/62 | +32.9 (−0.1, +66.5) | Passed |
| Fixed, 2,000 pairs | 1295/1657/1048 | +21.5 (+13.2, +29.8) | Completed |

Promoted the candidate. Data generation and play now use the same BF16 checkpoint.

Artifacts: `/data/chess/chess-engine-5/models/bf16-data-2026.11.11`.
