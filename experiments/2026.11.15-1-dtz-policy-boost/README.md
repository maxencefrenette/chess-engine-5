# DTZ policy mixing

Test whether Lc0-style DTZ policy supervision makes tablebase value rescoring useful.

Sweep `policy_mix` over 0, 0.2, 1/3, and 0.5 with `value_mix = 1`. Mix the original policy with uniform probability over moves selected by Lc0's DTZ-only winning-move ranking. Apply policy mixing to all selected training positions, independently of value masking.

Keep original value targets when recorded history contains a repetition since the last zeroing move. The `policy_mix = 0` control includes this correction. Held-out targets remain untouched.

Each nonzero `policy_mix` adjusted 19,918 policy targets. Value rescoring changed 1,668 labels.

| `policy_mix` | Policy loss | WDL loss | Total loss |
|---:|---:|---:|---:|
| 0 | 1.959113 | 0.739664 | 2.698778 |
| 0.2 | 1.966903 | 0.744479 | 2.711382 |
| 1/3 | 1.972556 | 0.741744 | 2.714300 |
| 0.5 | 1.989464 | 0.750433 | 2.739897 |

Timed runtime was 308.79–312.36s per run, with 3286.51s of charged generation time: 3595.30–3598.87s total budget. Runs with policy mixing also spent 146.16–146.53s preparing targets, excluded from the timed budget.

Selected `policy_mix = 0.2`, `value_mix = 1` for a gainer SPRT. It did not beat the control or baseline on held-out loss; the match tests whether DTZ guidance improves conversion despite poorer agreement with the original targets. Matched-speed passed, with 800 visits per move for both models.

Gainer SPRT [0, 20] nElo **failed** after **105 game pairs**.

- W/D/L **45/84/81**
- nElo **−76.9 (−125.8, −27.8)**, 95% CI

Kept the previous baseline; tablebase rescoring remains disabled.

Artifacts: `/data/chess/chess-engine-5/models/dtz-policy-2026.11.15`.
