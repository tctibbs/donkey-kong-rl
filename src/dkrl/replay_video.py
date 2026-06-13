"""Record a video of a Go-Explore trajectory replaying deterministically.

Renders the exact action sequence found by `explore` (open-loop, deterministic
sticky via the hash rule), proving the discovered score is real and
reproducible. This is the Go-Explore discovery result, distinct from a
stochastic-protocol policy evaluation.
"""

import pickle
from pathlib import Path

import ale_py
import gymnasium as gym
import imageio
import numpy as np
from loguru import logger

from dkrl.explore import _sticky_effective

gym.register_envs(ale_py)


def record_trajectory_video(
    trajectory_path: Path,
    out_path: Path,
    rank: int = 0,
) -> dict:
    """Render one explored trajectory to an MP4 with full emulator frames.

    Args:
        trajectory_path: Pickle written by `explore`.
        out_path: Output MP4 path.
        rank: Which trajectory by descending score (0 is best).

    Returns:
        Dict with the replayed score and output path.
    """
    with trajectory_path.open("rb") as f:
        payload = pickle.load(f)
    sticky = float(payload.get("sticky", 0.0))
    trajectories = sorted(payload["trajectories"], key=lambda t: t[0], reverse=True)
    score_target, actions = trajectories[rank]

    base = gym.make(
        "ALE/DonkeyKong-v5",
        frameskip=4,
        repeat_action_probability=0.0,
        full_action_space=True,
        render_mode="rgb_array",
    )
    base.reset(seed=payload["start_seed"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(out_path, fps=30, codec="libx264", quality=7)

    total = 0.0
    p_eff = 0
    for i, a in enumerate(actions):
        eff = _sticky_effective(int(a), p_eff, i, sticky)
        _, reward, terminated, truncated, _ = base.step(eff)
        p_eff = eff
        total += float(reward)
        writer.append_data(np.asarray(base.render()))
        if terminated or truncated:
            break
    base.close()
    writer.close()
    logger.info(f"trajectory video: target {score_target:.0f}, replayed {total:.0f}, {out_path}")
    return {"score_target": score_target, "score_replayed": total, "path": str(out_path)}
