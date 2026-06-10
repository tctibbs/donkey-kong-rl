"""Tests for config loading, serialization, and hashing."""

from pathlib import Path

from dkrl.config import ExperimentConfig


def test_yaml_round_trip(tmp_path: Path) -> None:
    """A config survives YAML serialization unchanged."""
    cfg = ExperimentConfig(name="t", seed=3, total_timesteps=1000)
    path = tmp_path / "cfg.yaml"
    cfg.to_yaml(path)
    loaded = ExperimentConfig.from_yaml(path)
    assert loaded == cfg


def test_hash_excludes_name_and_seed() -> None:
    """Multi-seed runs of one experiment share a config hash."""
    a = ExperimentConfig(name="a", seed=1)
    b = ExperimentConfig(name="b", seed=2)
    assert a.config_hash == b.config_hash


def test_hash_sensitive_to_hyperparameters() -> None:
    """Changing any hyperparameter changes the hash."""
    a = ExperimentConfig(name="a")
    b = a.model_copy(deep=True)
    b.ppo.learning_rate = 1e-4
    assert a.config_hash != b.config_hash


def test_hash_is_stable() -> None:
    """The hash is deterministic across instantiations."""
    assert ExperimentConfig(name="x").config_hash == ExperimentConfig(name="y").config_hash
