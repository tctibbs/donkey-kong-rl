# ADR 0001: CleanRL-style PPO on EnvPool as the base stack

Status: accepted. Date: 2026-06-10.

## Context

The project needs a reproducible, hackable, single-GPU pipeline for
`ALE/DonkeyKong-v5` on a GTX 1080 Ti (Pascal) with 12 CPU cores. Atari
simulation is CPU-bound, so environment throughput, not GPU speed, dominates
wall-clock time.

## Decision

- Adapt CleanRL's `ppo_atari_envpool.py` into a modular package rather than
  using a framework like Stable-Baselines3 or a JAX stack.
- Use EnvPool (C++ vectorized ALE) for both training and evaluation envs.
- Use the PyTorch path, not JAX/XLA: Pascal predates good XLA support and the
  PyTorch EnvPool variant is the most reliable on this card.
- Pin `torch==2.5.1+cu121`. PyTorch dropped Pascal (sm_61) kernels from
  CUDA 12.8 wheel builds, so do not bump to cu128+ wheels.
- Python is pinned to 3.11 because EnvPool 0.8.4 ships wheels only up to 3.11.

## Consequences

- PPO first establishes the pipeline and a trainable baseline (Phase 1), then
  sparse-reward methods build on the same env and eval modules (Phase 2).
- CleanRL is citable and its hyperparameters are community-validated, giving
  the paper a defensible starting point.
- Measured on this box: the smoke config (16 envs, 8 threads) trains at about
  3,000 agent steps per second on CUDA. Throughput tuning happens in Phase 1.
