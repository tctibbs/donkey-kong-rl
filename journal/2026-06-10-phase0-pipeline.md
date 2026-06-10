# Phase 0: pipeline up, smoke test green, random baseline measured

One-line summary: the end-to-end pipeline (EnvPool, PPO, loguru, ledger,
checkpoints, metrics CSV) works on the 1080 Ti, and the random baseline is
122.0 mean over 100 protocol episodes.

## What happened

- Verified hardware: GTX 1080 Ti visible, torch 2.5.1+cu121 reports CUDA, and
  a matmul runs on device. EnvPool 0.8.4 lists `DonkeyKong-v5`.
- EnvPool API facts confirmed by direct inspection (these matter for stats):
  - Full action space is Discrete(18).
  - Observations are uint8, shape (num_envs, 4, 84, 84).
  - `info["reward"]` carries the raw (unclipped) per-step reward even when
    `reward_clip=True`; the step reward is the clipped one.
  - `info["terminated"]` marks real game-over; the step's `terminated` array
    also fires on life loss when `episodic_life=True`. `EpisodeStats` uses
    the info dict so logged returns are true episode scores.
  - DonkeyKong reports `lives=2` at start (three lives total, two in reserve).
- Smoke run (`configs/ppo_smoke.yaml`, 65,536 steps, 16 envs, 8 threads):
  trained in 21 s at about 3,100 steps per second, wrote checkpoint, metrics
  CSV, and ledger row. Run id ppo_smoke_s1_20260610_235531, config hash
  f70777f2767f.
- Random baseline (100 episodes, Machado protocol, seed 1000): mean 122.0,
  std 92.3, median 100, max 400. Ledger row kind=random_baseline.

## Lessons

- The host's NVIDIA userspace had been upgraded to 535.309 while the loaded
  kernel module was 535.288, so any new CUDA context failed. Fixed by
  installing `nvidia-dkms-535` (which builds matching modules for the running
  kernel) and reloading the module once the GPU was free. Documented in
  CLAUDE.md hardware notes.
- "Survives screen 1" baseline definition still needed: random play scores up
  to 400 without clearing the screen. A concrete operational definition (for
  example, mean episode length and reaching the 5,000-ish score a first-screen
  clear yields) should be fixed early in Phase 1.
