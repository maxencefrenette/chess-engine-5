# Data generation batching

Improve self-play throughput with immediate game replacement, four leaves per game using virtual loss, and a concurrency sweep.

Complete-game sweep: 2,000 games per arm in one shard, 64 visits per move, four leaves per game. Times exclude startup and writing the shard.

| Concurrent games | Positions | Generation (s) | Positions/s |
|---:|---:|---:|---:|
| 128 | 366,851 | 243.49 | 1,507 |
| 250 | 362,504 | 255.05 | 1,421 |
| 500 | 362,006 | 278.94 | 1,298 |
| 1,000 | 362,212 | 294.86 | 1,228 |

Production-depth search-loop probes used 1,200 visits, 20 seconds of warmup and 60 seconds of measurement. These measure evaluated leaves, not completed-game throughput.

| Concurrent games | Leaves/game | Evaluated leaves/s | Repeat |
|---:|---:|---:|---:|
| 250 | 1 | 80,284 | 77,929 |
| 128 | 4 | 84,425 | 82,885 |
| 250 | 4 | 80,936 | — |
| 500 | 4 | 78,598 | — |
| 1,000 | 4 | 79,774 | — |

Selected **128 games and four leaves per game**. The winner and one-leaf control were repeated in reverse order; their mean rates differ by **5.8%**.

Tests verify seeded game contents across concurrency settings with a deterministic evaluator and exact visit totals. Actual BF16 games differed across batch sizes. Four-leaf search changes the targets; this benchmark does not establish a playing-strength gain.

Artifacts: `/data/chess/chess-engine-5/models/datagen-batching-2026.11.13`.
