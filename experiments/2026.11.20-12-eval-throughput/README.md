# Evaluation throughput

Measure how much overlapping MCTS search with GPU inference speeds up `eval-model` on the RTX 5070. Each fixed match used the same 128 opening pairs (seed 42), the baseline and its preceding checkpoint, 128 active games and 800 visits per move for each checkpoint.

| Runner | Time for 128 pairs | Mean GPU utilization |
|---|---:|---:|
| Original | 265 s | 38% |
| Shared queue, two workers | 165 / 168 s | 70% / 68% |
| Shared queue, four workers | 177 s | 68% |

Evaluation now shares datagen’s CPU worker queue. Each group searches games that need the same checkpoint as its first active game. The GPU combines ready groups by checkpoint. With 128 active games, the median steady-state batch was 128 positions per model. The time-matching benchmark uses one full worker group.

Of the 256 seeded games, 207 followed the original move sequence in the traced run. The original and queued runs played 31,425 and 30,938 plies respectively. The older checkpoint scored 74/57/125 W/D/L in the queued run, −78.2 nElo (95% CI −120.9, −35.4).

## Active games

One fixed match per setting used the same 500 opening pairs at 800 visits per move. Times are cumulative and count pairs in SPRT scoring order. Peak host RAM was measured separately over 75 seconds of play.

| Active games | First 10 pairs | First 100 pairs | All 500 pairs | Peak host RAM |
|---:|---:|---:|---:|---:|
| 256 | 3m30s | 5m50s | 8m06s | 1.33 GiB |
| 512 | 6m16s | 7m38s | 7m49s | 2.17 GiB |

Set the evaluation default to 256 active games. It delivered the first 100 pairs 1m48s earlier and took 17s longer to finish all 500.

Raw timings, GPU samples and per-game traces are in `/data/chess/chess-engine-5/models/eval-throughput-2026.11.20`.
