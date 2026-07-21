# Phase 0 random-opponent pilot — gate failed

The first full paired matrix used a frozen randomly initialized actor-critic as
the opponent. It completed three seeds per reward mode, 100 updates per run,
52,428,800 environment steps per run, and 314,572,800 steps total.

| Seed | Shaped final win rate | Sparse final win rate | Shaped first >=90% | Sparse first >=90% |
|---:|---:|---:|---:|---:|
| 2027 | 1.00 | 1.00 | 5,242,880 | 5,242,880 |
| 2028 | 1.00 | 1.00 | 5,242,880 | 10,485,760 |
| 2029 | 1.00 | 1.00 | 5,242,880 | 5,242,880 |

Shaped reward was perfect at every evaluation point. Sparse reward also reached
1.00 in every seed and was already above 0.94 at its first evaluation in two of
three seeds. The requested shaped-versus-sparse gap was therefore not clear or
reproducible; the benchmark saturated too early.

Decision: **do not start return redistribution**. Replace the weak opponent with
the versioned `scripted_v1` frozen mixture and repeat the paired Phase 0 matrix.
