# Reinvest the training speedup

Spend the release-build savings on more unique positions while keeping epochs fixed, check non-regression before promotion.

| Training positions | Epochs | Optimizer steps | Policy loss | WDL loss | Total loss |
|---:|---:|---:|---:|---:|---:|
| 250,301 | 15.4147 | 30,151 | 2.2478 | 0.8206 | 3.0684 |
| 290,000 | 15.4147 | 34,929 | 2.2447 | 0.8192 | 3.0639 |

The candidate spent 1,793.88 weighted seconds: 287.60s timed runtime plus 0.3 × 5,020.93s estimated generation. Setup took 1.27s and was excluded.

- W/D/L 648/938/614 over 2,200 games
- Observed Elo +5.4
- Non-regression verdict Passed (retrospective evaluation at [−20, 0], LLR +10.1173)
- Visits per move: 799 candidate, 800 baseline

The existing match was re-evaluated under non-regression bounds after the experiment policy changed. The candidate is now the baseline, with 290,000 positions and unchanged epochs carried forward.

Artifacts: `/data/chess/chess-engine-5/models/speedups-2026.11.10/rebudgeted`; match log: `../rebudgeted-gainer.log` relative to that directory.
