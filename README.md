# Chess Engine 5

Yet another attempt at a chess engine. The goal of this project is to make a lc0-like engine with 100% original code and 100% original training data. Many ideas and techniques are taken from other engines.

## Methods

* MCTS with neural nets ([AlphaZero](https://arxiv.org/abs/1712.01815), [Leela Chess Zero](https://lczero.org/))

## Evaluation Methodology

In order to facilitate high experiment throughput, the data generation and neural net training are capped at a combined budget of 90min on my RTX 5070 system. The data generation process uses the previous best network to generate data. This allows studying this as a pretraining problem instead of a reinforcement learning problem, which is easier. This repository uses normalized elo everywhere. SPRT testing is run using a reimplementation of the ([fishtest](https://github.com/official-stockfish/fishtest)) methodology. Gainer bounds are set to [0,20] and non-regression bounds are [-20,0].
