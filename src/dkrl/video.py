"""Record evaluation videos of a trained agent via the ALE gymnasium path.

EnvPool cannot render, so video uses ale-py with preprocessing that mirrors
the training observations (grayscale, 84x84, frame skip 4, frame stack 4)
while writing every raw emulator frame to an MP4 at 60 fps.
"""

from pathlib import Path
from typing import Any, Protocol

import ale_py
import gymnasium as gym
import imageio
import numpy as np
import torch
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
from loguru import logger

from dkrl.config import ExperimentConfig
from dkrl.evaluate import load_policy

gym.register_envs(ale_py)


class VideoWriter(Protocol):
    """Minimal protocol for an imageio frame writer."""

    def append_data(self, frame: np.ndarray, /) -> None:
        """Append one RGB frame."""
        ...


class FrameSink(gym.Wrapper):
    """Wrapper that streams every rendered frame to a video writer."""

    def __init__(self, env: gym.Env, writer: VideoWriter) -> None:
        """Wrap ``env`` and append each step's render output to ``writer``.

        Args:
            env: Environment with ``render_mode="rgb_array"``.
            writer: An imageio writer with an ``append_data`` method.
        """
        super().__init__(env)
        self.writer = writer

    def step(self, action: int) -> tuple:
        """Step the env and record the rendered frame."""
        out = self.env.step(action)
        self.writer.append_data(np.asarray(self.env.render()))
        return out

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple:
        """Reset the env and record the first frame."""
        out = self.env.reset(seed=seed, options=options)
        self.writer.append_data(np.asarray(self.env.render()))
        return out


def record_video(
    checkpoint_path: Path,
    out_path: Path,
    episodes: int = 1,
    seed: int = 7,
) -> dict:
    """Roll out a checkpoint with rendering and write an MP4.

    The behavior protocol matches evaluation (sticky actions p=0.25, full
    action set, no-op starts) but runs a single env through ale-py because
    EnvPool does not expose rendering.

    Args:
        checkpoint_path: Path to a training checkpoint.
        out_path: Output MP4 path.
        episodes: Number of episodes to record into one file.
        seed: Seed for the env and action sampling.

    Returns:
        Dict with per-episode returns and the output path.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ExperimentConfig.model_validate(ckpt["config"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = load_policy(ckpt, cfg, device)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(out_path, fps=60, codec="libx264", quality=7)
    base = gym.make(
        f"ALE/{cfg.env.env_id}",
        frameskip=1,
        repeat_action_probability=cfg.eval.sticky_action_prob,
        full_action_space=cfg.eval.full_action_space,
        render_mode="rgb_array",
    )
    env: gym.Env = FrameSink(base, writer)
    env = AtariPreprocessing(
        env,
        noop_max=cfg.eval.noop_max,
        frame_skip=cfg.env.frame_skip,
        screen_size=cfg.env.img_size,
        grayscale_obs=True,
    )
    env = FrameStackObservation(env, cfg.env.frame_stack)

    torch.manual_seed(seed)
    returns = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_return, steps = 0.0, 0
        while True:
            x = torch.from_numpy(np.asarray(obs)[None]).to(device)
            action = policy(x)
            obs, reward, terminated, truncated, _ = env.step(int(action.item()))
            ep_return += float(reward)
            steps += 1
            if terminated or truncated or steps >= cfg.eval.max_episode_steps:
                break
        returns.append(ep_return)
        logger.info(f"video episode {ep + 1}/{episodes}: return={ep_return:.0f} steps={steps}")
    env.close()
    writer.close()
    logger.info(f"video written to {out_path}")
    return {"returns": returns, "path": str(out_path)}
