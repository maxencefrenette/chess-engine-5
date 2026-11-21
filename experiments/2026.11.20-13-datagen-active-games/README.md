# Active games in self-play

Choose the number of active games that generates the most saved positions per second on the RTX 5070. Each setting ran one complete 500-game shard with seed 42, 800 visits per move, four leaves per game and two search workers. Generation time includes finishing the last games in the shard.

| Active games | Saved positions | Generation (s) | Positions/s |
|---:|---:|---:|---:|
| 128 | 65,945 | 254.47 | 259.1 |
| 256 | 65,596 | 236.50 | 277.4 |
| 500 | 67,401 | 253.80 | 265.6 |

Set the default to **256 active games**, which generated 7.0% more positions per second than 128. All three shards contained 500 game IDs with consecutive plies and exactly 800 visits per position. With BF16 inference, 485 of 500 games at 256 followed the same moves as at 128, and 499 had the same outcome.

Shards and manifests are in `/data/chess/chess-engine-5/models/datagen-active-chuzzvvw`.
