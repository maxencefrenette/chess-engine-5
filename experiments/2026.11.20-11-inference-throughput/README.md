# Inference throughput during play

Find where self-play and evaluation spend time, then measure faster ways to run the same BF16 checkpoint on the RTX 5070.

## Bottleneck

Self-play spent 59.4% of a warmed 5,000-round search window outside the synchronous evaluator. It already averaged 507 of 512 possible leaves per call. Across a complete 256-game shard, including the tail, 53.8% of search time was outside the evaluator and GPU utilization averaged 35.5%. The 128-pair match between distinct checkpoints spent 48.7% outside the evaluator and averaged 37.5% GPU utilization. Runs used 128 active games, four leaves per game and 800 visits per move.

## Completed workloads

| Self-play, 256 games | Generation time | Saved positions/s |
|---|---:|---:|
| Original | 281.0 / 282.1 s | 116.6 / 116.1 |
| Eight workers in lockstep, optimized inference | 199.5 s | 164.2 |
| Two-worker queue, optimized inference | 134.8 / 138.3 s | 241.2 / 236.2 |

The new default averaged **2.05× the original throughput** and 77.5% GPU utilization. The GPU takes whatever is ready, up to 512 leaves. It hit a cached graph on 76.2% of all calls in the traced queue run, including startup and the shard tail.

With batch order held fixed, all moves, policies, root values and outcomes matched on 32,751 positions. Queued batches can have different BF16 results at the same seed. Across the two new-default shards, 16 and 13 games changed move sequence relative to the control. All 65,174 saved positions had exactly 800 visits.

The same 128 opening pairs took 280.1 and 280.5 seconds originally, then 265.3 and 263.1 seconds with optimized inference (5.7% less wall time). All four fixed matches gave the older checkpoint 74/61/121 W/D/L against the current baseline, −73.7 nElo (95% CI −116.4, −31.0), with pair counts [31, 26, 39, 23, 9].

## Shared CUDA inference

| Implementation | 251 boards | 256 boards | 508 boards | 512 boards |
|---|---:|---:|---:|---:|
| Original | 1.151 ms | 1.084 ms | 2.121 ms | 2.042 ms |
| Upload packed move IDs | 1.136 ms | 1.066 ms | 2.082 ms | 2.004 ms |
| Also score legal moves with tensor cores | 1.126 ms | 1.057 ms | 2.025 ms | 1.988 ms |

Times are median warmed end-to-end inference latencies on sampled self-play positions. Packed moves use one upload of two bytes per padded move, down from three uploads totaling twelve bytes. The width-32 BF16 policy head took 0.096 ms versus 0.117 ms for dense scoring at batch 512. All 101,734 legal logits agreed across 4,096 sampled positions. A separate check on the training run’s held-out games matched all 86,152 legal logits from 3,524 positions. A scalar reduction differed on one logit and was slower than the tensor-core kernel.

Replaying the original exact batch shapes after 4,000 warmup rounds, the share of calls hitting a cached CUDA graph rose from 15.1% to 58.3% in evaluation and from 40.7% to 58.0% in self-play. The cache admits up to eight frequent irregular shapes per model. Over the same 5,000 match shapes, inference averaged 1.130 ms with the original cache, 1.128 ms with two extra shapes and 1.104 ms with eight. The corresponding self-play latencies were 2.055, 2.035 and 2.015 ms.

## CPU scheduling

| Search workers | Lockstep, leaves/s | Shared queue, leaves/s |
|---:|---:|---:|
| 1 | 87,743 | 135,971 |
| 2 | 79,598 | 220,918 |
| 4 | 116,976 | 212,170 |
| 8 | 127,279 | 188,504 |

Both sweeps used packed move uploads. The lockstep figures cover 5,000 rounds after 4,000 warmup rounds. The queue figures cover 2.5 million leaves after 2 million warmup leaves. Each queue worker owns two groups of games so it can search one while the GPU evaluates the other. Two workers averaged 236 leaves per GPU call.

Raw traces, benchmark sources and manifests are in `/data/chess/chess-engine-5/models/inference-2026.11.20`.
