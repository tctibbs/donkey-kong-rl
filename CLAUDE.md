# CLAUDE.md

Persistent rules for this repository. These survive across sessions and override defaults.

## Project

Research codebase studying how far a single-GPU RL setup can push `ALE/DonkeyKong-v5`,
and which design choices drive the results. The output is a white paper backed by a
reproducible experiment suite. Treat "I can learn what mattered" as a first-class
requirement, equal to raw score.

## Commit rules

- Conventional Commits only: `feat`, `fix`, `docs`, `refactor`, `perf`, `test`,
  `build`, `ci`, `chore`. Imperative subject, under 72 characters. Body only when the
  why is not obvious.
- One logical change per commit. No wip commits on main.
- NEVER include AI attribution: no Co-Authored-By trailers, no "Generated with" lines.
  `.claude/settings.json` enforces this; do not remove it.
- Commit messages describe code changes, never run metrics. Results belong in
  `results/ledger.jsonl` and `journal/`.
- Work on short-lived feature branches, merge to `main` via PR with a
  conventional-commit title. Tag milestones.

## Engineering standards

- Toolchain is Astral only: `uv` for env and deps, `ruff` for lint and format,
  `ty` for type checking. All three must pass clean before a change is done:
  `uv run ruff check . && uv run ruff format --check . && uv run ty check`.
- Every dependency is pinned in `pyproject.toml` and locked in `uv.lock`.
- Logging via `loguru` only, with the run id bound to every line. No `print`.
- All configuration is Pydantic v2 models in `src/dkrl/config.py`, loadable from YAML,
  hashed for traceability. If a value could be ablated, it is a config field.
- Google-style docstrings on every public function and class. Full type hints.
- Validate at system boundaries only (config load, env creation, CLI). No speculative
  error handling.
- No em dashes anywhere: code, comments, docs, or prose. Use periods, commas,
  or parentheses.
- Deterministic seeding. Git SHA and config hash stamped into every ledger row.

## Public-ready expectation

The repo must always be ready to flip public with zero cleanup:

- No secrets, ever. `WANDB_API_KEY` lives in `.env` (gitignored) or the environment.
- No large binaries in history: checkpoints, videos, TensorBoard runs, and wandb dirs
  are gitignored. Share artifacts via GitHub Releases, decided deliberately.
- README, LICENSE, .gitignore stay current. CI must stay green.

## Memory system

- Experiment journal under `journal/`: one lesson per file, one-line summary at top.
- Design decisions as ADRs under `docs/adr/`.
- Read these before repeating work. Update rather than duplicate. Delete wrong notes.

## Hardware notes

- This box: GTX 1080 Ti (Pascal, sm_61), 12 CPU cores, 31 GB RAM. Atari simulation is
  CPU-bound; EnvPool is the throughput lever.
- PyTorch is pinned to a CUDA build that still ships Pascal kernels. Do not bump to a
  cu128+ build (Pascal support was dropped there).
- If `nvidia-smi` reports a driver/library version mismatch after a driver package
  upgrade, the fix is to free the GPU, then reload: `sudo rmmod nvidia_uvm nvidia_drm
  nvidia_modeset nvidia && sudo modprobe nvidia nvidia_uvm nvidia_drm`. The DKMS
  package `nvidia-dkms-535` keeps the module in sync with userspace.
