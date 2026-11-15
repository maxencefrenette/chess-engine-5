# Soft tablebase value rescoring

Test whether mixing game outcomes with tablebase WDL targets improves on hard rescoring.

`value_mix` is the tablebase fraction: `target = (1 - value_mix) × game_result + value_mix × tablebase_result`. `policy_mix` similarly specifies the fraction assigned to uniform probability over the selected DTZ moves. Retain game outcomes when repetition history makes rescoring unsafe.

| `value_mix` | `policy_mix` | Policy loss | WDL loss | Total loss |
|---:|---:|---:|---:|---:|
| 0 | 0 | 1.959161 | 0.705553 | 2.664713 |
| 0.1 | 0 | 1.960060 | 0.709133 | 2.669193 |
| 0.25 | 0 | 1.959437 | 0.713745 | 2.673182 |
| 0.5 | 0 | 1.959853 | 0.719223 | 2.679076 |
| 0.1 | 0.2 | 1.966880 | 0.709696 | 2.676577 |
| 0.25 | 0.2 | 1.967169 | 0.712936 | 2.680105 |
| 0.5 | 0.2 | 1.967252 | 0.729298 | 2.696549 |

The disabled-rescoring control reproduced the baseline's policy and WDL losses exactly. None of the rescored settings improved held-out total loss. Selected `value_mix = 0.1`, `policy_mix = 0` for a gainer match to check playing strength.

Timed runtime: 309.69–313.41s per run. Generation charge: 3286.51s. Total budget: 3596.20–3599.93s.

DTZ policy preparation added setup work: runs with `policy_mix = 0.2` spent 146.34–146.64s in setup, outside the timed budget.

Matched-speed passed; both models used 800 visits per move. Gainer SPRT [0, 20] nElo **failed** after **161 game pairs**.

- W/D/L **80/129/113**
- nElo **−44.0 (−83.1, −4.6)**, 95% CI

Kept the previous baseline; rescoring remains disabled.

Artifacts: `/data/chess/chess-engine-5/models/soft-rescoring-2026.11.15`.
