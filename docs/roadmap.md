# V2 GPU-first roadmap

## Completed foundation

1. Frozen V1 action, simulator, observation, reward, and termination contract.
2. Double-precision scalar oracle with CPU/GPU trajectory parity tests.
3. Device-resident SoA simulation, observations, masks, actions, rewards, reset, and summaries.
4. Device-resident actor-critic, rollouts, GAE, PPO, Adam, target-KL stopping, and numerical tests.
5. Checksummed atomic checkpoints and deterministic, byte-exact resume.
6. Side-balanced training, seeded randomized starts, fair draws, and distinct held-out evaluation.
7. GPU-native 13-feature visual-policy training and native checkpoint live inference.
8. Full training-loop benchmark, portable CUDA architecture configuration, allocation cleanup,
   consistent warnings, and Compute Sanitizer workflow.
9. Multi-seed Phase 0 launcher and automatic per-side/per-style statistical report.

## Promotion gates

### Phase 0 - controlled baselines (complete)

Run shaped and sparse visual policies with identical seeds, compute, randomized starts, training
opponents, and held-out evaluation. A valid report needs at least three seeds; five are preferred.
The old same-opponent result is archived in `docs/phase0_results.md` and cannot satisfy this gate.

The completed five-seed visual matrix (`2027` through `2031`) produced a median held-out final win
rate of 0.672 for both shaped and sparse reward. Shaped reward had higher normalized win-rate AUC
(0.649 vs. 0.568), but did not establish a final-performance advantage. Full median/IQR, side,
style, behavior, wall-time, and PPO metrics are in the
[generated report](phase0_heldout_v2_visual_report.md).

### Phase 1 - return redistribution (gated/no-go from current evidence)

Enter only if shaped reward has a reproducible held-out advantage over sparse reward, with better
win-rate AUC/final win rate and no side/style robustness regression. Otherwise this phase is a
correct no-go.

### Phase 2 - population-based tuning

Enter only after a Phase 1 candidate exists. Tune learning rate, entropy, rollout/minibatch sizes,
and conservative reward coefficients. Select on held-out outcomes, never optimized shaped return.

### Phase 3 - historical league/self-play

Enter only after Phase 2. Maintain current, historical, exploitative, and scripted opponents;
promote snapshots through fixed held-out gates and retain history to detect forgetting.

## Signal separation

| Signal | Purpose |
|---|---|
| Optimization | PPO loss input: shaped or sparse reward |
| Selection | Held-out win rate, side/style robustness, exploitability |
| Audit | Sparse outcomes, behavior metrics, parity, resume, sanitizer tests |

## GPU invariants

- No production loop over environments on CPU.
- No observation, mask, action, reward, done, or rollout tensor crosses PCIe in the PPO hot loop.
- Aggregate evaluation crosses once per batch; full state downloads are checkpoint/debug only.
- The deployable policy uses the 13-feature visual contract; privileged policies are labeled.
- Benchmark the complete rollout-plus-update loop on the exact final build and target GPU.
