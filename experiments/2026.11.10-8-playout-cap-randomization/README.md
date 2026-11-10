# Playout cap randomization

Test whether more games with fewer retained rows per game improve playing strength at the same generation charge, inspired by [KataGo’s results in Go](https://arxiv.org/abs/1902.10565).

Generated 3,000 games and retained 276,932 full-search rows. Each move independently used 1,200 visits with probability 0.5 and 400 otherwise. The charge for each retained row includes its share of the fast moves.

| Recipe | Training positions | Training games | Policy loss | WDL loss |
|---|---:|---:|---:|---:|
| 1,200 visits on every move | 250,301 | 1,800 | 2.2945 | 0.8192 |
| Random 400 or 1,200 visits | 184,192 | 2,697 | 2.3013 | 0.8254 |

Both scores use the original 1,200-visit held-out games.
Combined held-out loss was lower on the baseline, but SPRT was still run in case loss wasn't representative of playing strength.

## Gainer SPRT

- W/D/L 17/30/53 over 100 games
- Observed Elo -130.9
- SPRT verdict Failed
- Visits per move 802 candidate, 800 baseline

Run artifacts in `/data/chess/chess-engine-5/models/supervision-2026.11.10.c0H3Yd/playout-cap-randomization`.
