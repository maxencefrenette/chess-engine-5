# Cheap-to-deep curriculum

Test whether a 400-visit phase followed by a 1,200-visit phase improves strength over training on 1,200-visit data alone.

Trained for 6,459 steps on 60,000 cheap 400-visits positions, then 24,256 steps on 225,290 high quality 1200-visits positions. All held-out loss comes from the same 1,200-visit games.

| Recipe | 400-visit positions seen | 400-visit epochs | 1,200-visit positions seen | 1,200-visit epochs | Policy loss | WDL loss |
|---|---:|---:|---:|---:|---:|---:|
| 1,200 visits only | 0 | — | 3,858,323 | 15.41 | 2.2945 | 0.8192 |
| 400 → 1,200 visits | 826,336 | 13.77 | 3,103,234 | 13.77 | 2.2973 | 0.8155 |

## Gainer SPRT

- W/D/L 230/233/245 over 708 games
- Observed Elo -7.4
- SPRT verdict Failed
- Visits per move 800 candidate, 800 baseline

Run artifacts in `/data/chess/chess-engine-5/models/supervision-2026.11.10.c0H3Yd/cheap-to-deep-vs-1200`.
