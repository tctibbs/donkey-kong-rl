"""Pydantic v2 configuration schemas for all experiments.

Every meaningful hyperparameter and design choice is a typed field here so it
can be ablated one variable at a time. Configs load from YAML, serialize back
to YAML, and hash deterministically so every run is traceable.
"""

import hashlib
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class EnvConfig(BaseModel):
    """EnvPool Atari environment construction parameters.

    Defaults follow the Machado et al. (2018) ALE recommendations where they
    apply to training: sticky actions and the full action set. Episodic life
    and reward clipping are training conveniences and are ablatable.
    """

    env_id: str = "DonkeyKong-v5"
    num_envs: int = Field(default=64, ge=1)
    num_threads: int = Field(default=12, ge=0, description="0 lets EnvPool pick")
    img_size: int = 84
    frame_stack: int = 4
    frame_skip: int = 4
    noop_max: int = 30
    sticky_action_prob: float = Field(default=0.25, ge=0.0, le=1.0)
    full_action_space: bool = True
    episodic_life: bool = True
    reward_transform: Literal["clip", "scale", "none"] = Field(
        default="clip",
        description="clip: EnvPool [-1,1] clipping; scale: raw times reward_scale; none: raw",
    )
    reward_scale: float = Field(default=0.01, gt=0.0)
    fire_reset: bool = True
    max_episode_steps: int = Field(
        default=27_000, description="Agent steps; 108k frames at skip 4"
    )


class PPOConfig(BaseModel):
    """PPO hyperparameters, following CleanRL ppo_atari_envpool defaults."""

    learning_rate: float = 2.5e-4
    anneal_lr: bool = True
    num_steps: int = Field(default=128, description="Rollout length per env per update")
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.1
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None


class RNDConfig(BaseModel):
    """Random Network Distillation hyperparameters (Burda et al. 2018).

    Only used when ``algo`` is ``ppo_rnd``. Defaults follow CleanRL
    ppo_rnd_envpool.
    """

    int_coef: float = 1.0
    ext_coef: float = 2.0
    int_gamma: float = Field(default=0.99, description="Intrinsic-return discount")
    update_proportion: float = Field(
        default=0.25, description="Fraction of samples for predictor loss"
    )
    feature_dim: int = 512
    obs_norm_init_rollouts: int = Field(
        default=50, description="Random-policy rollouts to warm up obs normalization"
    )


class SILConfig(BaseModel):
    """Self-Imitation Learning (Oh et al. 2018) hyperparameters."""

    init_from: str | None = Field(
        default=None, description="Feedforward checkpoint to warm-start the policy"
    )
    seed_trajectories: list[str] = Field(
        default_factory=list, description="Go-Explore trajectory pkls to seed the SIL buffer"
    )
    seed_self_demos: list[str] = Field(
        default_factory=list,
        description="Harvested self-demo pkls (the agent's own episodes) to seed the demo buffer",
    )
    buffer_size: int = 100_000
    demo_buffer_size: int = Field(default=60_000, description="Permanent non-evicting demo region")
    demo_fraction: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Fraction of each SIL batch drawn from demos"
    )
    updates_per_iter: int = Field(default=4, ge=0)
    batch_size: int = 512
    vf_coef: float = 0.01
    min_return: float = Field(
        default=0.0, description="Only buffer episodes above this raw return"
    )


class LSTMConfig(BaseModel):
    """Recurrent-policy hyperparameters (used when algo is ppo_lstm)."""

    hidden_size: int = 128
    distill_teacher: str | None = Field(
        default=None, description="Feedforward checkpoint to distill into the LSTM before PPO"
    )
    distill_chunks: int = Field(
        default=400, description="Teacher rollout chunks for distillation pretraining"
    )


class RobustifyConfig(BaseModel):
    """Backward-curriculum restore-reset training (Go-Explore phase 2)."""

    bank_path: str = "data/state_bank.pkl"
    board_filter: int | None = Field(
        default=None, description="0 barrels, 1 rivets, None all stages"
    )
    init_from: str | None = Field(
        default=None, description="Checkpoint to warm-start the policy (recurrent continue-train)"
    )
    recurrent: bool = Field(
        default=False, description="Use the LSTM agent for restore-reset training"
    )
    start_reset_prob: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of resets that start a real game (preserves start competence)",
    )
    start_window: float = Field(
        default=0.15, gt=0.0, le=1.0, description="Initial fraction of nearest-to-reward states"
    )
    schedule_frac: float = Field(
        default=0.6, gt=0.0, le=1.0, description="Training fraction over which the window widens"
    )
    max_episode_steps: int = 2000


class BCConfig(BaseModel):
    """Behavior-cloning warm-start from scripted demonstrations."""

    enabled: bool = False
    demo_path: str = "data/demos_screen1.npz"
    epochs: int = Field(default=10, ge=0, description="0 skips pretraining (aux loss only)")
    batch_size: int = Field(default=512, ge=1)
    learning_rate: float = 2.5e-4
    aux_coef: float = Field(
        default=0.0,
        ge=0.0,
        description="Kickstarting: BC cross-entropy added to the PPO loss; 0 disables",
    )
    aux_anneal_updates: int = Field(
        default=300, ge=1, description="Updates over which aux_coef anneals linearly to the floor"
    )
    aux_coef_floor: float = Field(
        default=0.0,
        ge=0.0,
        description="Permanent BC pressure after annealing; keeps clears from washing out",
    )


class EvalConfig(BaseModel):
    """Evaluation protocol, locked to the Machado et al. recommendations.

    Sticky actions p=0.25, full action set, 108k-frame cap, no-op starts,
    no episodic-life trick, raw (unclipped) rewards.
    """

    episodes: int = 100
    num_envs: int = 25
    sticky_action_prob: float = 0.25
    full_action_space: bool = True
    noop_max: int = 30
    max_episode_steps: int = 27_000
    seed: int = 1000
    clear_reward_threshold: float = Field(
        default=1500.0,
        description="Single-step raw reward marking a screen clear (ADR 0002)",
    )


class ExperimentConfig(BaseModel):
    """Root config for a training run."""

    name: str
    seed: int = 1
    total_timesteps: int = 10_000_000
    algo: Literal["ppo", "ppo_rnd", "ppo_robustify", "ppo_lstm", "ppo_sil"] = "ppo"
    env: EnvConfig = EnvConfig()
    ppo: PPOConfig = PPOConfig()
    rnd: RNDConfig = RNDConfig()
    bc: BCConfig = BCConfig()
    robustify: RobustifyConfig = RobustifyConfig()
    lstm: LSTMConfig = LSTMConfig()
    sil: SILConfig = SILConfig()
    eval: EvalConfig = EvalConfig()
    cuda: bool = True
    torch_deterministic: bool = True
    track: bool = True
    wandb_project: str = "donkey-kong-rl"
    wandb_entity: str | None = None
    checkpoint_interval_updates: int = Field(default=0, description="0 disables periodic saves")

    @property
    def config_hash(self) -> str:
        """Deterministic 12-hex-char hash of the config.

        Excludes `name` and `seed` so that multi-seed runs of the same
        experiment share a hash and group together in the ledger.
        """
        payload = self.model_dump_json(exclude={"name", "seed"})
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    @classmethod
    def from_yaml(cls, path: Path) -> "ExperimentConfig":
        """Load and validate a config from a YAML file."""
        with path.open() as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    def to_yaml(self, path: Path) -> None:
        """Serialize the config to a YAML file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            yaml.safe_dump(self.model_dump(mode="json"), f, sort_keys=False)
