# Tekken-lite V2

V2 is a parity-first, high-throughput C++ port of the Python Tekken-lite V1
simulator. The first milestone deliberately adds no new fighting mechanics.
It freezes the trainable action order, observation layout, reset state,
configuration defaults, move data, termination behavior, rewards, and step
metrics from V1 commit `7347212d036fcd6212fcf81864f6b2c96df0a524`.

This branch contains only V2 source. The complete original Python project is
preserved separately on the [`v1` branch](https://github.com/MrWatermeloooon/tekken8-agent/tree/v1).

## Build and test on Windows

```powershell
cmake -S . -B build -A x64
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

## Benchmark optimized CPU throughput

```powershell
build/Release/t8_v2_benchmark.exe --threads 1 --envs 256 --steps 20000
build/Release/t8_v2_benchmark.exe --threads 16 --envs 256 --steps 20000
```

The benchmark reports decision steps and simulated frames per second. CPU
scaling is measured before any CUDA or structure-of-arrays work is considered.

## Contract guard

`contracts/v1_contract.json` is the machine-readable snapshot of the V1
boundary. The C++ parity tests cover deterministic reset, privileged and visual
observations, neutral stepping, simultaneous hits, throw breaks, action masks,
and timeout rewards.

## Deliberate non-goals for this milestone

- No checkpoint format migration yet.
- No PPO implementation yet.
- No CUDA path yet.
- No deeper mechanics or observation changes yet.

Those begin only after deterministic simulator parity and multicore CPU
throughput are established.
