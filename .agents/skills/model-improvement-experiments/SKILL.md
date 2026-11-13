---
name: model-improvement-experiments
description: Plan, run, and interpret Chess Engine 5 experiments on training data, model architecture, training recipes, and model inference.
---

# Scope

This skill covers data generation and labeling, data selection and sampling, model architecture, training recipes, and model inference. Engine and search changes are out of scope. All experiments should compare a candidate against the baseline with the goal of getting a measurable nElo gain.

## Multiple experiments

Run experiments sequentially. Finish each experiment’s evaluation and promotion decision before starting the next candidate in order to start from an updated baseline.

## Training budget

Every training run is subject to a budget of 90min. It is defined as the time to train the model + the time required to generate the data that went into training the model. The budget used is reported by `mise run train`. Held-out data is excluded from the calculation. The timed runtime excludes data loading and setup and includes held-out evaluation and checkpoint saving. All arms of an experiment must be roughly budget-matched by being within 5% of the 90min budget.

## Sweeps

Treat a sweep as one experiment. Start all variants from the same baseline settings and select only the most promising candidate for the match.

## Experiment Types

Assign each experiment exactly one category based on its deliberate intervention. Speed and nElo changes are outcomes and do not determine the category. Supporting implementation work and budget reallocation follow the primary intervention.

### Data generation

Changes how self-play trajectories and search labels are produced, including the teacher, search depth, and move sampling.

Required evidence:
* Train a new model with the new data
* `mise run eval-model -- gainer --matched-speed "$candidate_dir"`

### Training data usage

Changes which existing examples supervise each output, or when and how often they are presented. This includes position selection, epochs, curriculum order, and target masking.

Required evidence: `mise run eval-model -- gainer --matched-speed "$candidate_dir"`

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

Changes how predictions and targets produce parameter updates, including target definitions, objectives, optimizers, learning-rate schedules, batch size.

Intermediate evidence: Held-out loss is a useful metric to determine if a candidate deserves a full SPRT test.
Required evidence: `mise run eval-model -- gainer --matched-speed "$candidate_dir"`

### Inference approximation

Changes how a trained model's predictions are approximated during play, including inference precision and approximate operators. The goal is that the extra positions evaluated benefit more that the loss of accuracy incured.

Examples:
* Using an inference kernel that slightly changes numerical outputs

Require evidence: `mise run eval-model -- gainer "$candidate_dir"`

### Search and playing behavior

Changes how predictions become moves during play. Search behavior during data generation belongs to Data generation. This category covers search changes during evaluation or other play and is outside this skill's current scope.

Out of scope for now since we don't have a good testing setup for this.

### Behavior-preserving implementation

Changes how the same computation is executed while preserving its intended results. This includes optimizations to training, inference, data generation, and match execution. A batching change belongs here only if it preserves each game's search behavior.

Required evidence:
* A benchmark showing a speedup
* A correctness comparison

### Experimental measurement

Changes how systems are measured or compared, including statistical tests, benchmarking, time matching, and budget definitions. These changes only require a theoretical justification as to why they're good for pursuing the goal of maximizing nElo gains per day. Add a new baseline if needed. Mention any relevant data that led to this decision.

### Combined intervention

Deliberately combines two or more independent changes from Data generation through Search and playing behavior. Use this category only when the changes are separate experimental choices. Supporting code changes and adjustments needed to match the budget do not make an experiment combined. Reuse checkpoint weights for inference-only experiments when possible and verify that each side uses its intended inference path. Use your judgement to determine what evidence is required for a candidate to replace the baseline.

## Updates

After a successful baseline match, set `baseline_dir` in `config/eval-model.toml` to the candidate checkpoint directory. Carry forward winning settings in `config/training.toml` and `config/datagen.toml` where applicable.

## Report

After finishing an experiment, regardless of if it's successful or not, write an experiment report in the experiments folder. Each experiment report is a folder of the form `experiments/{date}-{n}-{experiment-name}` where date is the date in `YYYY.MM.DD` format, n is the number of the experiment within a day and experiment-name is the name of the experiment. The folder should contain a README.md of the main experiment report.

An experiment report must include the following:
* The 1-sentence goal of the experiment
* Timed training runtime, estimated training-data generation time, and weighted total
* [optional] svg charts, embedded in the README.md

If a baseline match was run, include:
* The candidate's wins, draws, and losses (W/D/L)
* Observed nElo with its 95% confidence interval
* The SPRT verdict (passed, failed, or inconclusive)
