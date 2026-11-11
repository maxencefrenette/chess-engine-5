# Deferred loss readbacks

Test whether copying detached losses to the CPU every ten steps speeds up training.

Same fixed one-epoch workload as experiment 14; deferred/original/original/deferred order.

| Implementation | Training time, two runs (s) |
|---|---|
| Original release | 16.34, 16.42 |
| Deferred readbacks | 16.08, 16.19 |

The measured reduction was 1.5%, with overlap against earlier release timings (16.14–16.31s). Checkpoint tensors and all CSV fields except elapsed time were exactly equal. Not retained: the small gain does not justify extra accumulation logic without stronger evidence. No match was run.

Artifacts, including the tested patch: `/data/chess/chess-engine-5/models/speedups-2026.11.10`.
