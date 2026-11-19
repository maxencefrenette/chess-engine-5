# Value sampling and loss weight

Sweep the fraction of positions that receive the mixed WDL target and the weight of that loss. At each weight, match the new candidate with the lowest unweighted held-out policy plus outcome WDL loss.

| Value weight | Sample rate | Policy | Outcome WDL | Total loss | Training (s) | Generation (s) | Budget (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.03 | 1/8 (baseline) | 1.83947 | 0.66112 | 2.50059 | 358.53 | 3228.22 | 3586.75 |
| 0.03 | 1/4 | 1.84011 | 0.65949 | 2.49961 | 354.66 | 3228.22 | 3582.87 |
| 0.03 | 1/2 | 1.83972 | 0.65990 | 2.49962 | 356.98 | 3228.22 | 3585.19 |
| 0.1 | 1/8 | 1.84003 | 0.65659 | 2.49662 | 355.08 | 3228.22 | 3583.30 |
| 0.1 | 1/4 | 1.84116 | 0.65601 | 2.49717 | 355.54 | 3228.22 | 3583.75 |
| 0.1 | 1/2 | 1.84428 | 0.65427 | 2.49855 | 357.11 | 3228.22 | 3585.33 |
| 0.3 | 1/8 | 1.84285 | 0.65832 | 2.50117 | 354.42 | 3228.22 | 3582.64 |
| 0.3 | 1/4 | 1.85133 | 0.65198 | 2.50331 | 354.21 | 3228.22 | 3582.43 |
| 0.3 | 1/2 | 1.84079 | 0.65239 | 2.49318 | 353.79 | 3228.22 | 3582.01 |

Matched-speed gainers against the mixed WDL baseline.

| Value weight | Sample rate | W/D/L | nElo (95% CI) | Verdict |
|---:|---:|---|---|---|
| 0.03 | 1/4 | 338/386/354 | -3.5 (-25.2, +18.8) | failed |
| 0.1 | 1/8 | 741/899/728 | +5.7 (-9.3, +21.9) | failed |
| 0.3 | 1/2 | 244/248/186 | +34.1 (+6.3, +61.4) | passed |

Promoted weight 0.3 with a 1/2 sample rate. Its held-out WDL loss fell from 0.66112 to 0.65239 while policy loss rose from 1.83947 to 1.84079.

Artifacts in `/data/chess/chess-engine-5/models/root-wdl-2026.11.18/root-value-sampling-and-weight`.
