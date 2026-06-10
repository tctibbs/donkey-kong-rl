.PHONY: install lint format typecheck test check smoke baseline

install:
	uv sync

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run ty check

test:
	uv run pytest -q

check: lint typecheck test
	uv run ruff format --check .

smoke:
	uv run dkrl train --config configs/ppo_smoke.yaml --no-track

baseline:
	uv run dkrl random-baseline --episodes 100
