# CUDA graphs and compact board uploads

Reduce inference overhead with reusable CUDA graphs and GPU board encoding.

| Inference path | Batch-512 latency (ms) |
|---|---:|
| Original | 2.52 |
| Graphs with expanded CPU inputs | 2.52 |
| Compact uploads and GPU encoding | 2.10 |
| Both | 2.03 |

Three warmed 500-call repetitions per path. Board uploads shrink from 5,376 to 37 bytes per position. Most of the gain comes from compact uploads; graph replay adds a smaller benefit.

Graphs are cached for exact power-of-two batch sizes; irregular sizes use eager inference. Batch dimensions are never padded. The cache holds at most four shapes.

Complete-game comparison, in original/optimized/optimized/original order: 256 games per run, 64 visits, 128 concurrent games and four leaves per game.

| Path | Generation time (s), two runs | Combined positions/s |
|---|---|---:|
| Original | 32.457, 32.848 | 1,425 |
| Optimized | 29.481, 29.319 | 1,582 |

Throughput improved **11.1%**. Every run produced the same 46,516 positions; game IDs, snapshots, policy visit counts, played moves and outcomes matched exactly.

CUDA tests verify exact CPU/GPU encoding and exact eager/optimized predictions across changing inputs, batch sizes, move counts and cache eviction. Workspace tests and CPU/CUDA lint passed.

Artifacts: `/data/chess/chess-engine-5/models/cuda-inputs-2026.11.13`.
