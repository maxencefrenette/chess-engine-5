---
name: model-improvement-experiments
description: Plan, run, and interpret Chess Engine 5 experiments that change the model architecture or training recipe.
---

# Scope

This skill describes how to run and evaluate experiments that improve the model architecture or training recipes. Modifications to the training data or to the engine are out of scope.

## Evaluation

To determine whether an experiment is successful, the new model must be run against the old model using the following command:

```sh
mise run eval-model -- "$candidate_dir"
```

After a successful experiment, update the baseline.

```sh
mise run update-model -- "$candidate_dir"
```
