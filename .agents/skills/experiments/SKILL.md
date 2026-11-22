---
name: experiments
description: Plan, run, and interpret experiments on training data, model architecture, training recipes, model inference, and search in batched games.
---

# Scope

This skill covers data generation and labeling, data selection and sampling, model architecture, training recipes, model inference, and search in batched games. All experiments should compare a candidate against the baseline with the goal of getting a measurable nElo gain.

## Multiple experiments

Run experiments sequentially. Finish each experiment’s evaluation and promotion decision before starting the next candidate in order to start from an updated baseline.

## Training budget

Every training run is subject to a budget of 60min. It is defined as the time to train the model + the time required to generate the data that went into training the model. The budget used is reported by `mise run train`. Held-out data is excluded from the calculation. The timed runtime excludes data loading and setup and includes held-out evaluation and checkpoint saving. All training runs must be roughly budget-matched by being within 5% of the 60min budget.

## Sweeps

Treat a sweep as one experiment. Start all variants from the same baseline settings and select only the most promising candidate for the match.

## Experiment Types

Assign each experiment exactly one category based on its deliberate intervention. Speed and nElo changes are outcomes and do not determine the category. Supporting implementation work and budget reallocation follow the primary intervention.

### Data generation

Changes how self-play trajectories and search labels are produced, including the teacher, search depth, and move sampling.

Required evidence:
* Train a new model with the new data
* `mise run eval-model -- gainer "$candidate_dir" --matched-speed`

### Training data usage

Changes which existing examples supervise each output, or when and how often they are presented. This includes position selection, epochs, curriculum order, and target masking.

Required evidence: `mise run eval-model -- gainer "$candidate_dir" --matched-speed`

### Model architecture

Changes what the shape of the model, which impacts training and inference speed.

Examples:
* Model dimension
* Number of layers
* FFN width
* Training at a lower precision data type

Required evidence: `mise run eval-model -- gainer "$candidate_dir"`

### Model ablation

Same as model architecture changes, but seeks to simplify the model architecture without losing nElo.

Examples:
* Disabling smolgen

Required evidence: `mise run eval-model -- non-regression "$candidate_dir"`

### Learning rule

Changes how predictions and targets produce parameter updates, including optimizers, learning-rate schedules, batch size, adding auxiliary heads.

Intermediate evidence: Held-out loss is a useful metric to determine if a candidate deserves a full SPRT test.
Required evidence: `mise run eval-model -- gainer "$candidate_dir" --matched-speed`

### Training objective

Changes what the deployed policy or value head is trained to predict, or how training weights errors on those outputs. Examples include sharpening visit targets and changing the relative weights of policy and value loss. Multiple candidates may be run in a match against the baseline.

Required evidence: `mise run eval-model -- gainer "$candidate_dir" --matched-speed`

### Inference approximation

Changes how a trained model's predictions are approximated during play, including inference precision and approximate operators. The goal is that the extra positions evaluated benefit more that the loss of accuracy incured.

Examples:
* Using an inference kernel that slightly changes numerical outputs

Require evidence: `mise run eval-model -- gainer "$candidate_dir"`

### Search

Changes how search turns predictions into moves during a match, including the PUCT exploration constant and subtree reuse. Changes made only to produce training games belong to Data generation. Time management requires timed games and is outside the scope of `eval-search`.

Use `eval-search` for a gainer with the configured baseline checkpoint and equal total root visits for both players. The baseline always plays with the settings in `config/engine.toml`. The candidate starts with those settings and takes overrides through `--set candidate.KEY=VALUE`.

```sh
mise run eval-search -- gainer --set candidate.KEY=VALUE
```

### Behavior-preserving speedup

Changes how the same computation is executed while preserving its intended results. This includes optimizations to training, inference, data generation, and match execution. A batching change belongs here only if it preserves each game's search behavior.

Required evidence:
* A benchmark showing a speedup
* A correctness comparison

### Experimental measurement

Changes how systems are measured or compared, including statistical tests, benchmarking, time matching, and budget definitions. These changes only require a theoretical justification as to why they're good for pursuing the goal of maximizing nElo gains per day. Add a new baseline if needed. Mention any relevant data that led to this decision.

### Combined intervention

Deliberately combines two or more independent changes from Data generation through Search. Use this category only when the changes are separate experimental choices. Supporting code changes and adjustments needed to match the budget do not make an experiment combined. Use your judgement to determine what evidence is required for a candidate to replace the baseline.

## Updates

After a successful checkpoint match, set `baseline_dir` in `config/eval-model.toml` to the candidate checkpoint directory. Carry forward winning settings in `config/training.toml` and `config/datagen.toml` where applicable. After a successful search-only match, promote the winning engine settings in `config/engine.toml`.

## Report

After finishing an experiment, regardless of if it's successful or not, write an experiment report in the experiments folder. Each experiment report is a folder of the form `experiments/{date}-{n}-{experiment-name}` where date is the date in `YYYY.MM.DD` format, n is the number of the experiment within a day and experiment-name is the name of the experiment. The folder should contain a README.md of the main experiment report.

An experiment report must include the following:
* The 1-sentence goal of the experiment
* [optional] svg charts, embedded in the README.md

If a baseline match was run, include:
* The candidate's wins, draws, and losses (W/D/L)
* Observed nElo with its 95% confidence interval from the final match summary. When a gainer passes early, use its fixed-sample estimate over all pairs at the configured floor.
* The SPRT verdict (passed, failed, or inconclusive)
