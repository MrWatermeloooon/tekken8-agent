# Phase 0 five-seed result — no-go for redistribution

## Protocol

- Benchmark: frozen GPU `scripted_v1` eight-style mixture.
- Seeds: 2027, 2028, 2029, 2030, 2031.
- Reward modes: shaped and sparse terminal win/loss.
- Per run: 512 environments, 32-step rollouts, 100 updates, two PPO epochs,
  1,024-sample minibatches, and 1,638,400 environment steps.
- Evaluation: 256 episodes after every update using the same frozen benchmark
  and outcome-only win/loss/draw metric.
- Total: 10 runs and 16,384,000 environment steps.

## Results

| Seed | Reward | Mean first 10 evals | Mean full curve | Steps to >=90% | Mean final 10 evals |
|---:|---|---:|---:|---:|---:|
| 2027 | Shaped | 0.6000 | 0.9322 | 81,920 | 1.0000 |
| 2027 | Sparse | 0.1211 | 0.9065 | 212,992 | 1.0000 |
| 2028 | Shaped | 0.5688 | 0.9294 | 16,384 | 1.0000 |
| 2028 | Sparse | 0.1832 | 0.8742 | 262,144 | 1.0000 |
| 2029 | Shaped | 0.1000 | 0.8788 | 163,840 | 1.0000 |
| 2029 | Sparse | 0.4555 | 0.9455 | 49,152 | 1.0000 |
| 2030 | Shaped | 0.0000 | 0.8769 | 196,608 | 1.0000 |
| 2030 | Sparse | 0.5777 | 0.9214 | 311,296 | 1.0000 |
| 2031 | Shaped | 0.2000 | 0.8773 | 65,536 | 1.0000 |
| 2031 | Sparse | 0.1000 | 0.8573 | 49,152 | 1.0000 |

| Aggregate | Shaped median (IQR) | Sparse median (IQR) |
|---|---:|---:|
| Steps to >=90% | 81,920 (65,536–163,840) | 212,992 (49,152–262,144) |
| Mean full curve | 0.8788 (0.8773–0.9294) | 0.9065 (0.8742–0.9214) |
| Mean first 10 evals | 0.2000 (0.1000–0.5688) | 0.1832 (0.1211–0.4555) |
| Mean final 10 evals | 1.0000 | 1.0000 |

Shaped reached 90% sooner in three seeds and sparse did so in two. Full-curve
performance favored shaped in three seeds and sparse in two. The interquartile
ranges overlap, and both modes converge to perfect benchmark performance.

## Gate decision

The Phase 0 question was whether a clear, reproducible shaped-versus-sparse gap
exists. It does not. Therefore:

- learned return redistribution is **not justified and is not implemented**;
- PBT is not started because there is no validated Phase 1 system to tune;
- league self-play is not started because its preceding gates are unmet.

This is a successful no-go decision, not a failed implementation. V2 already
solves the current frozen benchmark from sparse terminal outcomes alone. Later
phases should be reopened only when a harder environment or benchmark produces
a measured sparse-credit-assignment failure.
