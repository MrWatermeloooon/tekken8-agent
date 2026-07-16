# Tekken-lite V2 — GPU-first

V2 is a CUDA-first, massively batched Tekken-lite environment for reinforcement
learning. The production simulator keeps states, observations, legal-action
masks, rewards, and terminal flags in GPU memory. One CUDA thread advances one
fight through a complete four-frame decision step.

The scalar C++ simulator is retained only as a deterministic correctness oracle.
It is not the V2 training runtime.

The complete Python V1 project is preserved on the
[`v1` branch](https://github.com/MrWatermeloooon/tekken8-agent/tree/v1). V2 is
self-contained: the frozen V1 contract and reusable move data needed by this
branch are copied under `contracts/` and `data/`.

## GPU architecture

- CUDA is mandatory for the V2 production target.
- Fight state uses structure-of-arrays storage in VRAM for coalesced warp access.
- Combat stepping, movement, hit detection, rewards, termination, observations,
  and action masks all execute in CUDA kernels.
- `step_device()` accepts GPU-resident `uint8` action tensors.
- `step_device_i64()` accepts the native `int64` output of common PyTorch/JAX
  argmax pipelines, avoiding an extra conversion or CPU copy.
- Output pointers from `device_view()` stay in VRAM and can be wrapped directly
  by the future PPO tensor layer.
- Float32 is used on the production path because consumer RTX cards are built
  for high FP32 throughput. GPU-vs-CPU semantic parity is checked with tight
  tolerances against the double-precision oracle.

## Build and test

Requirements: an NVIDIA GPU, CUDA Toolkit 13.1 or compatible, and CMake 3.24+.

```powershell
cmake -S . -B build-cuda -A x64
cmake --build build-cuda --config Release
ctest --test-dir build-cuda -C Release --output-on-failure
```

This machine is currently validated against an NVIDIA GeForce RTX 5070 Ti
(compute capability 12.0) with CUDA 13.1.

## Benchmark the device-resident hot loop

```powershell
build-cuda/Release/t8_v2_gpu_benchmark.exe --envs 262144 --steps 2000
```

The benchmark keeps actions and every simulator output on the GPU. It includes
full combat simulation, observation/mask generation, rewards, termination, and
automatic reset of completed environments.

## Contract and plan

`contracts/v1_contract.json` freezes V1's action order, observation layout,
configuration, moves, reset state, rewards, termination, and metrics at commit
`7347212d036fcd6212fcf81864f6b2c96df0a524`.

The implementation and training sequence is in `docs/roadmap.md`.
