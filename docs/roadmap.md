# V2 GPU-first roadmap

## Foundation gates

1. **Freeze V1 contracts — complete.** Actions, observations, config, moves,
   rewards, termination, and metrics are captured in `contracts/v1_contract.json`.
2. **Build a scalar oracle — complete.** The C++ scalar simulator reproduces
   deterministic V1 fixtures and exists only to detect behavior drift.
3. **Move the entire environment hot loop to CUDA — complete.** Device-resident
   SoA state, movement, combat, rewards, termination, observations, masks, and
   reset execute on the GPU. Both uint8 and native int64 policy actions are
   accepted without a host round trip.
4. **Validate and profile on target hardware — complete for the foundation.** GPU trace parity must pass before
   throughput numbers count. Nsight profiling will guide kernel fusion, launch
   amortization, occupancy, and memory-layout changes.
5. **Add the GPU PPO learner — complete.** Rollouts, GAE, minibatches,
   policy/value updates, action selection, and simulator stepping remain on one
   CUDA device. Checkpoints include Adam state and evaluation uses frozen sparse
   outcomes.

## Reward-lean training plan

### Phase 0 — controlled baselines (complete)

Train shaped-reward and sparse-reward baselines with identical seeds, compute,
opponent pools, and evaluation. Keep optimization, selection, and audit signals
separate so shaping cannot silently become the success metric.

The random-opponent pilot saturated; see `docs/phase0_pilot_random.md`. The
five-seed high-resolution rerun against `scripted_v1` also found no clear,
reproducible shaped-versus-sparse gap. See `docs/phase0_results.md`.

### Phase 1 — return redistribution (no-go: prerequisite gap absent)

Introduce return redistribution only after the baselines are stable. Compare it
against the same seed set and held-out opponents. Accept it only if sparse return,
win rate, exploitability, and robustness improve—not merely training reward.

Not implemented. Sparse PPO already reaches the same final benchmark result,
and the five-seed learning curves do not show a consistent gap to close.

### Phase 2 — population-based training (not entered)

Use PBT to tune learning rate, entropy, rollout/minibatch sizes, and conservative
reward coefficients. Selection uses held-out league performance rather than the
same shaped objective being optimized.

### Phase 3 — league training (not entered)

Maintain current, historical, exploitative, and scripted opponents. Promote
policies through fixed evaluation gates and retain snapshots to prevent forgetting.

## Signal separation

| Signal | Purpose |
|---|---|
| Optimization | Dense learning signal consumed by PPO |
| Selection | Held-out win rate, exploitability, and matchup robustness |
| Audit | Sparse task return, behavior metrics, and regression fixtures |

## Update hierarchy

Change one level at a time: simulator contract, then reward/redistribution, then
PPO hyperparameters, then population/league composition. Use at least three seeds
for iteration and five for promotion-quality comparisons.

## GPU rules

- No production simulator step may depend on a CPU loop over environments.
- No observation, mask, action, reward, or done tensor crosses PCIe in the PPO
  hot loop.
- CPU downloads are parity/debug APIs only.
- Prefer large persistent batches; benchmark several environment counts to find
  the throughput/latency knee for the 16 GB target GPU.
- Profile before splitting kernels or increasing precision. Float64 remains an
  oracle concern unless a concrete parity failure proves FP32 insufficient.
