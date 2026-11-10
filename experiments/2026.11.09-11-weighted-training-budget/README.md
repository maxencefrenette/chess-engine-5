# Weighted training budget

Define the training budget as timed training plus 0.1 times the generation cost of the training positions. Held-out generation is excluded. Under this accounting, the previous baseline cost 23 min 9 s. We set the new target to 20 min and measured the Elo change.

The new baseline trained for 7,160 steps instead of 20,429 on the same positions and spent 1,234.91 weighted seconds. The fixed match ran for 100 color-swapped games at 798 candidate visits and 800 old-baseline visits per move.

- W/D/L 24/29/47
- Observed Elo -81.4
