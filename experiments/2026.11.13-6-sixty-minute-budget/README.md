# 60-minute training budget

Reduce the budget from 65 to 60 minutes by using fewer training positions at 15 epochs, and measure the nElo loss.

| Budget | Training positions | Timed training (s) | Generation charge (s) | Total (s) |
|---|---:|---:|---:|---:|
| 65 minutes | 350,821 | 351.29 | 3547.74 | 3899.03 |
| 60 minutes | 323,849 | 318.86 | 3274.99 | 3593.85 |

Fixed comparison of 1,000 game pairs against the 65-minute checkpoint, at 800 visits per move for both models after matched-speed validation.

- W/D/L **487/793/720**
- nElo **-54.1 (-69.4, -38.9)**, 95% CI
- Fixed match completed

Set the 60-minute checkpoint as the new budget baseline.

Artifacts: `/data/chess/chess-engine-5/models/sixty-minute-baseline-2026.11.13`.
