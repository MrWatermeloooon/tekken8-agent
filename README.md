# Tekken-lite V2 - GPU-first PPO

V2 is a CUDA-first, massively batched Tekken-lite environment and PPO learner.
The production path keeps simulator state, both observation contracts, masks,
actions, rewards, rollout storage, GAE, and optimization in VRAM. The scalar C++
simulator exists only as a deterministic correctness oracle.

V1 remains on the [`v1` branch](https://github.com/MrWatermeloooon/tekken8-agent/tree/v1).
All V2 implementation and live-runtime files are on the `v2` branch; there is no
duplicated `v1/` or `v2/` source folder.

## What is implemented

- Side- and style-balanced GPU training against eight scripted training styles.
- A separate eight-style held-out V2 evaluation suite; it is never used for rollout collection.
- Fair timeout draws, randomized seeded starts, and P1/P2 breakdowns.
- Privileged 19-feature and screen-compatible 13-feature observations generated on GPU.
- CUDA actor-critic inference, rollout storage, fixed-order GAE, clipped PPO policy/value losses,
  entropy, gradient clipping, Adam, and target-KL early stopping.
- Atomic checksummed checkpoints plus exact trainer state; interrupted runs resume byte-for-byte.
- A native `.t8ppo` Torch CUDA live loader using the V1-compatible screen/controller stack.
- Full-loop and simulator-only throughput benchmarks, automated Phase 0 aggregation, and CUDA tests.

## Build and test

Requirements: Windows, an NVIDIA GPU, CUDA Toolkit 13.1 or compatible, Visual Studio 2022,
and CMake 3.24+.

```powershell
cmake -S . -B build -A x64 -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure
```

Architecture `120` is correct for the validated RTX 5070 Ti. The source default follows CMake's
[CUDA architecture model](https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html)
and emits native
images for Turing, Ampere, Ada, and Blackwell and can be overridden with
`-DCMAKE_CUDA_ARCHITECTURES=...` for another deployment target.

## Train a deployable visual policy

```powershell
build\Release\t8_v2_train.exe --smoke `
  --observation-mode visual --reward shaped --seed 2027
```

Use a unique `--run-dir` for each new run. Resume an interrupted run with the matching options:

```powershell
build\Release\t8_v2_train.exe --updates 100 `
  --observation-mode visual --reward shaped --seed 2027 `
  --resume runs\my_run\checkpoints\update_40.t8ppo
```

`--updates` is the final target update, not an additional count. V2 refuses artifact overwrite,
option drift, corrupt checkpoints, and metrics that do not end at the resume update.

## Benchmarks

```powershell
build\Release\t8_v2_gpu_benchmark.exe --envs 262144 --steps 2000
build\Release\t8_v2_training_benchmark.exe --envs 4096 --horizon 128 --updates 5
```

On the RTX 5070 Ti, the final visual build reached a five-run median of 1.95 million environment
decisions/s and 9.08 million PPO sample-visits/s with the command above. The simulator-only
benchmark median was 425.64 million decisions/s (1.70 billion simulated frames/s). These are local
throughput results, not portable performance guarantees; rerun the exact commands on each target
system.

## Live screen inference

Install the live dependencies with a CUDA-enabled PyTorch build selected from the
[official PyTorch installer](https://pytorch.org/get-started/locally/), then install this project:

```powershell
python -m pip install -e ".[live]"
Copy-Item config\live_screen.example.yaml config\live_screen.yaml
python scripts\live_vision_play.py --dry-run --agent v2 `
  --ppo-checkpoint runs\my_visual_run\checkpoints\update_100.t8ppo
```

Dry-run first. Controller output starts paused, defaults to CUDA inference, and requires an
integrity-protected 13-feature V2 checkpoint. Press the configured hotkey to enable output only
after capture, player side, and facing are confirmed.

The completed five-seed visual Phase 0 baseline found faster shaped-reward learning, but no final
held-out win-rate advantage, so return redistribution remains gated. See the
[generated report](docs/phase0_heldout_v2_visual_report.md),
[training details](docs/training.md), and [roadmap](docs/roadmap.md).
