# Joint MLP and transformer value logits

Test whether a small pooled transformer head improves WDL prediction by adding logits to the material MLP.

Used the baseline MLP width and added a 16-unit transformer head after mean pooling. Its final projection started at zero. Both branches trained jointly from scratch with one WDL loss.

| Recipe | Epochs | Policy loss | WDL loss |
|---|---:|---:|---:|
| Baseline | 15.41 | 2.2478 | 0.8206 |
| hybrid | 15.41 | 2.2857 | 1.5291 |

No candidate improved baseline held-out combined policy and WDL loss.

Training WDL loss reached 0.3766 at the last full epoch, while held-out WDL loss was 1.5291. The combined head still showed substantial value overfitting.

Run artifacts in `/data/chess/chess-engine-5/models/value-capacity-2026.11.10.Hzhknn/joint-value-logits`.
