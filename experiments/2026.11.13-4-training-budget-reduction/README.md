# Training budget reduction

Reduce the nominal training budget from 90 to **65 minutes** using measured production datagen throughput.

Generated 500 complete games at 1,200 visits, 128 concurrent games, and four leaves per game: **101,645 positions in 1,257.31s**, or **80.84 positions/s**. Timing includes the shard's final games.

| Component | Time (s) |
|---|---:|
| Generation estimate for 291,816 training positions | 3609.65 |
| Recorded baseline timed training runtime | 293.72 |
| Total, generation weight 1 | 3903.37 |

The estimate is **65m03s**, rounded to a **65-minute** nominal budget with the existing 5% tolerance. This uses one production-size shard; training runtime is from the baseline's recorded run.

Existing dataset generation charges retain their historical measured costs.

Artifacts: `/data/chess/chess-engine-5/models/budget-benchmark-2026.11.13`.
