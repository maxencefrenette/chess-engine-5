---
name: model-improvement-experiments
description: Plan, run, and interpret Chess Engine 5 experiments that change the model architecture or training recipe.
---

# Scope

This skill describes how to run and evaluate experiments that improve the model architecture or training recipes. Modifications to the training data or to the engine are out of scope.

## Sweeps

When sweeping multiple training configurations, only pick the most promising candidate to evaluate against the baseline in a match.

## Evaluation

The final test that determines whether an experiment is successful is a match against the baseline with SPRT testing. The new model must be run against the old model using the following command:

```sh
mise run eval-model -- gainer "$candidate_dir"
```

```sh
mise run eval-model -- non-regression "$candidate_dir"
```

After a successful experiment, update the baseline:

```sh
mise run update-model -- "$candidate_dir"
```

## Report

After finishing an experiment, regardless of if it's successful or not, write an experiment report in the experiments folder. Each experiment report is a folder of the form `experiments/{date}-{n}-{experiment-name}` where date is the date in `YYYY.MM.DD` format, n is the number of the experiment within a day and experiment-name is the name of the experiment. The folder should contain a README.md of the main experiment report.

An experiment report must include the following:
* The 1-sentence goal of the experiment
* [optional] svg charts, embedded in the README.md
* The candidate's wins, draws, and losses (W/D/L)
* Observed Elo as reported by `eval-model`
* The SPRT verdict (passed, failed, or inconclusive)
