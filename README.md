# Donkey Kong RL

A personal experiment in how far I can get a reinforcement learning agent to play
Atari Donkey Kong on one modest machine, and figuring out what actually makes a difference.

Donkey Kong (`ALE/DonkeyKong-v5`) is a hard game for RL. Rewards are sparse, episodes
run long, and it sits outside the usual Atari benchmark set, so there is not much to
compare against. I wanted to see how far I could push it, keep everything reproducible,
and leave an honest trail of what worked and what did not.

<p align="center">
  <img src="assets/donkey_kong_before_after.gif" alt="Left: a plain PPO agent farms the bottom of the first screen. Right: the same network trained with self-imitation climbs, clears, and loops the game.">
</p>

Left, a plain PPO agent (10M steps, no demonstrations) that converged to farming barrel jumps
at the bottom (4,027 points, never climbs). Right, the same network trained with self-imitation
from a demonstration buffer, now climbing, clearing, and looping both stages for a 44,600-point
run (its best games reach 59,500). Both under the stochastic evaluation protocol, shown at 4x
speed; the girders recolor each lap as the game ramps up.

## Results

Plain PPO farms 100-point barrel jumps and never climbs (mean 4,027, zero screen clears).
A chain of self-imitation rounds takes a trained closed-loop policy to a **3-seed mean of
32,420 (IQM 34,033), with a best episode of 59,500** under the stochastic protocol, about
4.6x the strongest no-demonstration agent. That is where this method tops out: the same
recipe collapses if pushed past ~30k, a training-stability ceiling documented in
[experiment 11](experiments/11-past-30k-and-the-sil-ceiling/writeup.md).

The whole arc, each experiment a self-contained write-up with its config, numbers, and
figures, is in **[experiments/](experiments/README.md)**.

## Hardware

One modest machine: an aging NVIDIA GTX 1080 Ti (Pascal) and 12 CPU cores. The binding
constraint is not the GPU. Atari stepping is CPU-bound, so the 12 cores set the pace:
throughput tops out around 5,000 environment steps per second, with
[EnvPool](https://github.com/sail-sg/envpool) doing the vectorized stepping. The
Nature-CNN policy is small enough that the old GPU is never the bottleneck; it just runs
the learner updates. The practical consequences run through the whole project: training
is gated by CPU env-throughput, runs go one at a time, and the levers that mattered were
algorithmic, not more compute.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and a CUDA-capable GPU with a driver
supporting CUDA 12.1.

```bash
git clone git@github.com:tctibbs/donkey-kong-rl.git
cd donkey-kong-rl
uv sync
```

Optional tracking with Weights and Biases: put `WANDB_API_KEY=...` in `.env`
(gitignored) or export it. Runs work without it (TensorBoard logs locally to `runs/`).

## Usage

Train (config-driven, every hyperparameter is a typed field):

```bash
uv run dkrl train --config configs/ppo_baseline.yaml
```

Evaluate a checkpoint under the standard ALE protocol (sticky actions p=0.25, full
action set, 108k-frame cap):

```bash
uv run dkrl evaluate --checkpoint checkpoints/<run_id>/final.pt --episodes 100
```

Measure the random-policy baseline:

```bash
uv run dkrl random-baseline --episodes 100
```

Every run writes a row to `results/ledger.jsonl` keyed by config hash, with git SHA,
seed, scores, frame count, and wall-clock, so results survive independently of any
tracking service.

## Reproducing results

Each ledger row records the exact config hash and git SHA. Check out that SHA, run the
config under `configs/` with the recorded seed, and scores should match within seed
noise. Headline results use 3 to 5 seeds and report mean, standard deviation, and
interquartile mean.

## Quality gates

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pytest
```

CI runs the same gates on every push and pull request.

## Repository layout

- `src/dkrl/` package: config schemas, env construction, agents, training loops,
  evaluation, ledger, plotting
- `configs/` YAML experiment configs
- `results/` results ledger (committed; small text)
- `journal/` experiment journal, one lesson per file
- `docs/adr/` architecture decision records

## License

MIT. See [LICENSE](LICENSE).
