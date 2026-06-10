# donkey-kong-rl

How far can a modern, single-GPU reinforcement learning setup push Atari Donkey Kong,
and which design choices actually move the needle?

`ALE/DonkeyKong-v5` is a sparse-reward, long-horizon game outside the canonical
Atari-57 benchmark set, so there is almost no published baseline. This repository is a
rigorous, reproducible study of that single game: a configuration-driven training
pipeline, a controlled ablation suite, and the artifacts a white paper needs (results
ledger, learning curves, evaluation videos, experiment journal).

## Status

Phase 0 (pipeline) in progress. No trained results yet. This section is updated as
results land.

## Hardware target

Single machine: NVIDIA GTX 1080 Ti (Pascal), 12 CPU cores. The Atari simulation is
CPU-bound; [EnvPool](https://github.com/sail-sg/envpool) provides the vectorized
environment throughput. The learner runs on the GPU.

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
