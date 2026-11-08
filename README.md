# Chess Engine 5

Yet another attempt at a chess engine. The goal of this project is to make a lc0-like engine with 100% original code and 100% original training data.

## Run a batched CPU match

```sh
mise exec -- cargo run --release --locked -p chess-engine -- match 8 800 10000 42
```

The optional arguments are concurrent games (default 8), visits per move
(800), maximum color-swapped pairs (10,000), and seed (42).

Both players use the current material evaluator with uniform priors. Each
active search can submit one leaf per round. The runner groups leaves by
player, evaluates each batch on CPU, and resumes the searches with the
returned values. Finished games are replaced until the game cap or the
stopping boundary.

Every game begins from the standard position. The first four plies sample by
visit counts and subsequent moves pick the most visited move. Each pair uses
a shared random stream and switches which player plays White.

The test uses a paired multinomial generalized SPRT at **[0, +10] logistic
Elo** for the candidate, with nominal 5% error rates and Wald boundaries
±log(19). It evaluates pairs in assigned order using five pair score bins
(0, ½, 1, 1½, and 2 candidate points). If the maximum pair count is reached
inside the boundaries, the result is inconclusive.
