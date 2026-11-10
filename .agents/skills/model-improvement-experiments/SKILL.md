---
name: model-improvement-experiments
description: Plan, run, and interpret Chess Engine 5 experiments that change the model architecture or training recipe.
---

# Scope

This skill describes how to run and evaluate experiments that improve the model architecture or training recipes. Modifications to the training data or to the engine are out of scope.

## Multiple experiments

Run experiments sequentially. Finish each experiment’s evaluation and promotion decision before training the next candidate. Use the current baseline’s architecture and training recipe as the starting settings for each experiment.

## Time budget

Every training run should aim to take 5 minutes. Use the timed runtime printed by `mise run train`, which starts after data loading and setup and includes held-out evaluation and checkpoint saving. Rerun training outside the 285–315 second window with an adjusted step count. When a new configuration changes the time per step, calibrate its step count to finish in about 5 minutes.

## Sweeps

Treat a sweep as one experiment. Train all variants from the same baseline recipe and select only the most promising candidate for the match.

## Evaluation

Choose one SPRT mode before running the baseline match, based on the experiment's goal.

Use `gainer` for experiments that aim to improve playing strength under a fixed training and inference budget.

* Model architecture improvements
* Model training recipe improvements
* Lower training cost, the model should be trained for more steps in the same wall-clock time budget, which should result in an improvement.
* Lower inference cost, the SPRT match will automatically assign more visits to the faster candidate, which should result in increased playing strength.

```sh
mise run eval-model -- gainer "$candidate_dir"
```

Use `non-regression` when the goal is to simplify the codebase, model, or training recipe while preserving playing strength.

* Codebase simplifications
* Ablations that simplify the model architecture
* Ablations that remove the need for a hyperparameter

```sh
mise run eval-model -- non-regression "$candidate_dir"
```


Update the baseline after a successful experiment.

```sh
mise run update-model -- "$candidate_dir"
```

## Report

After finishing an experiment, regardless of if it's successful or not, write an experiment report in the experiments folder. Each experiment report is a folder of the form `experiments/{date}-{n}-{experiment-name}` where date is the date in `YYYY.MM.DD` format, n is the number of the experiment within a day and experiment-name is the name of the experiment. The folder should contain a README.md of the main experiment report.

An experiment report must include the following:
* The 1-sentence goal of the experiment
* [optional] svg charts, embedded in the README.md

If a baseline match was run, include:
* The candidate's wins, draws, and losses (W/D/L)
* Observed Elo as reported by `eval-model`
* The SPRT verdict (passed, failed, or inconclusive)
