---
name: model-improvement-experiments
description: Plan, run, and interpret Chess Engine 5 experiments that change the model architecture or training recipe.
---

# Scope

This skill describes how to run and evaluate experiments that improve the model architecture or training recipes. Modifications to the training data or to the engine are out of scope.

## Multiple experiments

Run experiments sequentially. Finish each experiment’s evaluation and promotion decision before training the next candidate. Use the current baseline’s architecture and training recipe as the starting settings for each experiment.

## Training budget

Keep the budget fixed at 1,388.53 seconds, the cost of the one-layer baseline under this accounting. Use the training budget reported by `mise run train`. It adds timed runtime to 0.1 times the estimated generation time for the selected training positions, charging each position once across all epochs and excluding held-out generation. The timed runtime excludes data loading and setup and includes held-out evaluation and checkpoint saving.

Use a dataset with generation timings. Set `train_positions`, then calibrate `optimizer_steps` to fit the remaining budget. Rerun candidates whose reported budget differs from the target by more than 5%, adjusting the step count. Keep the same target after promoting a candidate.

## Sweeps

Treat a sweep as one experiment. Train all variants from the same baseline recipe and select only the most promising candidate for the match.

## Evaluation

Choose one SPRT mode before running the baseline match, based on the experiment's goal.

Use `gainer` for experiments that aim to improve playing strength under a fixed training and inference budget.

* Model architecture improvements
* Model training recipe improvements
* Lower training cost, which lets the model take more optimizer steps within the same training budget.
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
* Timed training runtime, estimated training-data generation time, and weighted total
* [optional] svg charts, embedded in the README.md

If a baseline match was run, include:
* The candidate's wins, draws, and losses (W/D/L)
* Observed Elo as reported by `eval-model`
* The SPRT verdict (passed, failed, or inconclusive)
