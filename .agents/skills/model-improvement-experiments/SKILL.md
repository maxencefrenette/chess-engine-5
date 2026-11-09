---
name: model-improvement-experiments
description: Plan, run, and interpret Chess Engine 5 experiments that change the model architecture or training recipe.
---

# Scope

This skill describes how to run and evaluate experiments that improve the model architecture or training recipes. Modifications to the training data or to the engine are out of scope.

## Evaluation

To determine whether an experiment is successful, the new model must be run against the old model using the following command:

```sh
mise run eval-model -- --set "baseline_dir=$baseline_dir" --set "candidate_dir=$candidate_dir"
```

Evaluation defaults to `config/eval-model.toml`. To use another file, pass
`--config PATH`. Repeat `--set KEY=VALUE` to override individual settings.
