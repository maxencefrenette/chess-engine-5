# Weight surprising policy targets

Give the highest fifth of training positions by KL(search visits ∥ generating network) additional policy weight, normalized to mean one. Teacher scoring time is charged to each run.

Baseline held-out total loss: 2.629102.

| Setting | Policy | WDL | Total loss | Runtime (s) | Generation (s) | Total budget (s) |
|---|---:|---:|---:|---:|---:|---:|
| surprise-1.5 | 1.945190 | 0.691045 | 2.636235 | 308.95 | 3286.51 | 3595.46 |
| surprise-2.0 | 1.949994 | 0.692899 | 2.642893 | 309.93 | 3286.51 | 3596.44 |

Held-out policy loss at playing precision, stratified by teacher surprise:

| Model | Top fifth | Other positions |
|---|---:|---:|
| Baseline | 2.379388 | 1.828703 |
| surprise-1.5 | 2.379114 | 1.836873 |
| surprise-2.0 | 2.385260 | 1.841408 |

At 1.5×, the small gain on the most surprising positions was outweighed by worse predictions elsewhere. At 2×, both groups regressed. Skipped the gainer.

Artifacts: `/data/chess/chess-engine-5/models/supervision-queue-2026.11.18/policy-surprise`.
