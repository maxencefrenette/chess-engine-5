# Random file reflection

Test whether reflecting 50% of eligible training positions improves generalization. Each exposure was sampled independently when both sides had lost castling rights. Held-out positions were scored in their original orientation. Of 45,045 held-out positions, 30,824 were eligible.

| Reflection | Policy loss | WDL loss | Total loss | Eligible policy | Eligible WDL | Timed runtime (s) | Generation (s) | Total budget (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0% | 1.938736 | 0.690366 | 2.629102 | 1.854934 | 0.621122 | 303.56 | 3286.51 | 3590.07 |
| 50% | 1.951677 | 0.694753 | 2.646431 | 1.871825 | 0.627993 | 309.50 | 3286.51 | 3596.01 |

Reflection increased total held-out loss by 0.017329. Both policy and WDL loss rose on eligible positions, so we skipped the gainer.

Artifacts: `/data/chess/chess-engine-5/models/reflection-2026.11.18`.
