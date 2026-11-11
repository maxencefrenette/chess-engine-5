# Reusable batch buffers

Test reusing host feature and move-index buffers in training and inference.

| Implementation | Training time, two one-epoch runs (s) | Inference at batch 256 (ms) | Batch 512 (ms) | Batch 1024 (ms) |
|---|---|---|---|---|
| Original release | 16.59, 16.08 | 1.502, 1.495 | 3.018, 3.017 | 6.388, 6.406 |
| Reused buffers | 15.83, 16.15 | 1.493, 1.500 | 3.018, 3.027 | 6.399, 6.416 |

Training used the same fixed workload as experiment 14. Inference used 500 warmed evaluations per size, in original/reused/reused/original order. Training weights and inference predictions were exactly equal, including alternating batch sizes 256, 17, 512, 1, 1024, 256 to check stale padding.

Not retained: approximately 2% lower mean training time with overlapping timings, and no measurable inference improvement. No match was run.

Artifacts, including the patch and benchmark binaries: `/data/chess/chess-engine-5/models/speedups-2026.11.10`.
