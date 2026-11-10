---
name: model-improvement-experiments
description: Plan, run, and interpret Chess Engine 5 experiments on training data, model architecture, training recipes, and model inference.
---

# Scope

This skill covers data generation and labeling, data selection and sampling, model architecture, training recipes, and model inference. Engine and search changes are out of scope. All experiments should compare a candidate against the baseline with the goal of getting a measurable Elo gain.

## Multiple experiments

Run experiments sequentially. Finish each experiment’s evaluation and promotion decision before starting the next candidate in order to start from an updated baseline.

## Training budget

Every training run is subject to a budget of 30min. It is defined as the time to train the model + 0.3 times the time required to generate the data that went into training the model. The budget used is reported by `mise run train`. Held-out data is excluded from the calculation. The timed runtime excludes data loading and setup and includes held-out evaluation and checkpoint saving. In order to compare apples to apples, keep all training runs within 5% of the budget.

## Sweeps

Treat a sweep as one experiment. Start all variants from the same baseline settings and select only the most promising candidate for the match.

## Experiment Types

Note: an experiment might fit into multiple categories.

### Data experiments

Data generation experiments are experiments that seek to improve the data generation process. If the experiment is successful, updates should be made to datagen.toml and the dataset field in training.toml. If the new data results in a change in the training budget, adjust hyperparameters to keep the training budget constant.

### Training/datagen time tradeoffs

If an experiment trades off datagen time budget for training time budget, report train time and datagen time for all arms of the experiment, otherwise do not report those values.

### Training speedup/slowdowns

This includes changes to the model architecture or to training kernels. The budget saved needs to be spent in other hyperparameters to stay close to the budget.

### Inference speedup/slowdown

This includes changes to the model architecture or to inference kernels. The evaluator benchmarks both models and scales candidate visits up or down relative to the baseline. For an inference-only change, reuse the baseline checkpoint when possible and verify that each side uses its intended inference path. A candidate model can win despite not having as low of a validation loss as the baseline.

### Speed-neutral changes

This includes changes to hyperparameters such as learning rate, initialization scheme. Validation loss might be a useful early indicator of whether it's even worth it to run the full SPRT test.

## Evaluation

Choose one SPRT mode before running the baseline match, based on the experiment's goal.

Use `gainer` for experiments that aim to improve playing strength under a fixed training and inference budget.

```sh
mise run eval-model -- gainer "$candidate_dir"
```

Use `non-regression` when the goal is simplification, such as model ablations or code simplification.

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
