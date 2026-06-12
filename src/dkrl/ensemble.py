"""Stage-expert ensemble: route observations to per-screen expert policies.

The barrels and rivets screens are visually distinct, so a tiny linear
classifier on downsampled pixels routes each observation to the right
expert. The composite (router plus experts) is a single observation-to-action
policy, fully protocol-legal for evaluation.
"""

from pathlib import Path
from typing import Any, cast

import ale_py
import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
from loguru import logger
from torch import nn

gym.register_envs(ale_py)

RAM_BOARD = 16


def collect_router_data(
    samples: int = 3000,
    seed: int = 5000,
    bank_path: Path = Path("data/state_bank.pkl"),
) -> tuple[np.ndarray, np.ndarray]:
    """Collect (frame, board) pairs by perturbed rollouts from bank states.

    Restoring bank states guarantees both stages appear; short random
    rollouts after each restore add visual variety (sprite positions).
    """
    import pickle

    from dkrl.robustify import BankEntry

    with bank_path.open("rb") as f:
        bank: list[BankEntry] = pickle.load(f)
    base = gym.make(
        "ALE/DonkeyKong-v5",
        frameskip=1,
        repeat_action_probability=0.0,
        full_action_space=True,
    )
    env = AtariPreprocessing(base, noop_max=0, frame_skip=4, screen_size=84, grayscale_obs=True)
    env = FrameStackObservation(env, 4)
    ale = cast(Any, base.unwrapped).ale
    rng = np.random.default_rng(seed)
    env.reset(seed=0)

    xs: list[np.ndarray] = []
    ys: list[int] = []
    while len(xs) < samples:
        entry = bank[int(rng.integers(0, len(bank)))]
        ale.restoreSystemState(pickle.loads(entry.state_blob))
        obs, _, terminated, truncated, _ = env.step(0)
        for _ in range(int(rng.integers(1, 12))):
            if terminated or truncated:
                env.reset(seed=0)
                break
            xs.append(np.asarray(obs, dtype=np.uint8)[-1])
            ys.append(int(ale.getRAM()[RAM_BOARD]))
            obs, _, terminated, truncated, _ = env.step(int(rng.integers(0, 18)))
    env.close()
    return np.stack(xs[:samples]), np.asarray(ys[:samples], dtype=np.int64)


class StageRouter(nn.Module):
    """Tiny linear classifier on a downsampled latest frame."""

    def __init__(self) -> None:
        """Build the linear head over a 21x21 downsample."""
        super().__init__()
        self.linear = nn.Linear(21 * 21, 2)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Classify the latest frame of a (N, 4, 84, 84) uint8 batch."""
        x = frames[:, -1].float() / 255.0
        x = torch.nn.functional.avg_pool2d(x.unsqueeze(1), 4).flatten(1)
        return self.linear(x)


def train_router(out_path: Path, samples: int = 3000, seed: int = 5000) -> dict:
    """Train and save the stage router; report holdout accuracy."""
    xs, ys = collect_router_data(samples, seed)
    frames = torch.from_numpy(xs)[:, None].repeat(1, 4, 1, 1)
    labels = torch.from_numpy(ys)
    n = len(labels)
    split = int(n * 0.8)
    perm = torch.randperm(n)
    tr, te = perm[:split], perm[split:]
    router = StageRouter()
    opt = torch.optim.Adam(router.parameters(), lr=1e-2)
    ce = nn.CrossEntropyLoss()
    for _ in range(200):
        logits = router(frames[tr])
        loss = ce(logits, labels[tr])
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = float((router(frames[te]).argmax(1) == labels[te]).float().mean())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(router.state_dict(), out_path)
    counts = {int(k): int(v) for k, v in zip(*np.unique(ys, return_counts=True), strict=True)}
    result = {"holdout_accuracy": acc, "samples": n, "class_counts": counts, "path": str(out_path)}
    logger.info(f"router trained: {result}")
    return result


def load_ensemble_policy(
    barrels_checkpoint: Path,
    rivets_checkpoint: Path,
    router_path: Path,
    device: torch.device,
) -> Any:  # noqa: ANN401
    """Build a batched policy that routes per-observation to stage experts."""
    from collections.abc import Callable

    from dkrl.config import ExperimentConfig
    from dkrl.evaluate import load_policy

    def load(path: Path) -> Callable[[torch.Tensor], torch.Tensor]:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ExperimentConfig.model_validate(ckpt["config"])
        return load_policy(ckpt, cfg, device)

    barrels = load(barrels_checkpoint)
    rivets = load(rivets_checkpoint)
    router = StageRouter().to(device)
    router.load_state_dict(torch.load(router_path, map_location=device, weights_only=True))
    router.eval()

    def policy(obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            stage = router(obs).argmax(1)
            a_b = barrels(obs)
            a_r = rivets(obs)
        return torch.where(stage == 0, a_b, a_r)

    return policy
