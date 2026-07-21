# Phase 0 report: phase0_heldout_v2_visual

All policy selection numbers use the held-out V2 suite, balanced across P1/P2 and eight styles.
Values are median [Q1, Q3] across seeds.

| Metric | Shaped | Sparse |
|---|---:|---:|
| Final win rate | 0.672 [0.664, 0.676] | 0.672 [0.664, 0.680] |
| Normalized win-rate AUC | 0.649 [0.637, 0.656] | 0.568 [0.555, 0.624] |
| P1 win rate | 0.672 [0.672, 0.672] | 0.672 [0.672, 0.672] |
| P2 win rate | 0.656 [0.648, 0.688] | 0.656 [0.648, 0.695] |
| Draw rate | 0.328 [0.320, 0.336] | 0.328 [0.320, 0.336] |
| Timeout rate | 0.391 [0.387, 0.395] | 0.391 [0.387, 0.391] |
| Mean damage dealt | 112.219 [112.008, 112.359] | 112.359 [112.219, 112.641] |
| Mean damage taken | 7.887 [7.625, 7.906] | 7.887 [7.625, 7.906] |
| Wall time (seconds) | 32.322 [31.774, 32.747] | 32.201 [31.163, 32.238] |
| Final approximate KL | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| Final entropy | 0.166 [0.099, 0.169] | 0.012 [0.010, 0.012] |
| Final gradient norm | 34.007 [32.652, 34.814] | 0.040 [0.031, 0.040] |
| Final value loss | 107.456 [105.352, 124.906] | 0.002 [0.001, 0.002] |

## Held-out style win rates

| Style | Shaped | Sparse |
|---:|---:|---:|
| 0 | 0.375 [0.312, 0.438] | 0.375 [0.312, 0.438] |
| 1 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| 2 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| 3 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| 4 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| 5 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| 6 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| 7 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |

## Gate decision

No promotion-quality shaped-reward advantage was established; keep return redistribution gated.
