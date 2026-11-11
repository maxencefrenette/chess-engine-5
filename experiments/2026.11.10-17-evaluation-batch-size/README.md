# Evaluation batch size

Increase evaluation throughput by batching more independent games while preserving four leaves per game.

Benchmark the same 256 seeded games through 32 plies, at fixed 800 visits for both sides. Includes search, inference, move selection, and replacement of completed games. The benchmark bypasses visit calibration and statistical stopping; its truncated games are not an Elo test.

| Concurrent games | Runtime (s) | Reverse-order repeat (s) |
|---|---:|---:|
| 64 | 78.09 | 79.84 |
| 128 | 74.30 | 74.67 |
| 256 | 76.52 | — |

All 256 per-game move sequences were exactly identical across all five runs. Accepted: evaluation defaults to 128 concurrent games. The improvement is approximately 5% on this workload; full-game throughput may differ. The maximum calibration batch is now 512 (128 × 4).

No training or model-strength match was needed. Artifacts: `/data/chess/chess-engine-5/models/speedups-2026.11.10`.
