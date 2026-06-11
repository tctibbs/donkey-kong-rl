# donkey-kong-rl

How far can a modern, single-GPU reinforcement learning setup push Atari Donkey Kong,
and which design choices actually move the needle?

`ALE/DonkeyKong-v5` is a sparse-reward, long-horizon game outside the canonical
Atari-57 benchmark set, so there is almost no published baseline. This repository is a
rigorous, reproducible study of that single game: a configuration-driven training
pipeline, a controlled ablation suite, and the artifacts a white paper needs (results
ledger, learning curves, evaluation videos, experiment journal).

## Status

- Phase 0 (pipeline) and Phase 1 (PPO baseline) complete (tags `v0.1.0-pipeline`,
  `v0.2.0-ppo-baseline`).
- Random baseline: mean 122 (100 protocol episodes). Plain PPO at 10M steps: mean
  4,027 but a 0.00 screen-1 clear rate (it farms barrel jumps; see the journal).
- Phase 2 complete: a systematic ladder of interventions (reward transforms, RND,
  entropy, action sets, behavior cloning variants) shows generic exploration never
  produces a single screen clear. The winning recipe is scripted-demonstration
  kickstarting: BC warm-start plus an annealed-to-a-floor BC auxiliary loss with
  two-screen demos. Best agent: mean 5,269, median 5,500, 93 percent screen-1
  clear rate over 100 protocol episodes.
- Phase 3 (ablations): the BC pressure floor is the load-bearing component
  (removing it collapses clearing to zero; inverted-U dose response peaking at
  0.15); the reward transform is irrelevant once behavior is demonstrated
  (clip vs scale within seed noise across 3 seeds each).
- Phase 4 (final agent): the production recipe at 25M steps, 3 seeds, pooled
  over 300 protocol episodes: mean 6,884 (95 percent CI 6,724 to 7,027), IQM
  7,071, median 7,100, screen-1 clear rate 0.97. That is 56x random and 71
  percent above the best no-demonstration agent. Draft methods and results:
  `docs/paper/methods-and-results.md`.

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
