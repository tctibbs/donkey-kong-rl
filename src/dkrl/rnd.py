"""Random Network Distillation components, after CleanRL ppo_rnd_envpool.

The intrinsic reward is the predictor's error against a frozen random target
network on the (normalized) latest observation frame. Observation and
intrinsic-reward normalization follow Burda et al. (2018).
"""

import numpy as np
import torch
from torch import nn

from dkrl.agents.ppo import layer_init


class RunningMeanStd:
    """Track running mean and variance with Welford-style batch updates."""

    def __init__(self, shape: tuple[int, ...] = ()) -> None:
        """Initialize statistics for arrays of the given shape."""
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, x: np.ndarray) -> None:
        """Fold a batch of samples (axis 0) into the running statistics."""
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        self.mean = new_mean
        self.var = m2 / tot_count
        self.count = tot_count


class RewardForwardFilter:
    """Discounted running return used to scale intrinsic rewards."""

    def __init__(self, gamma: float) -> None:
        """Initialize with the intrinsic discount factor."""
        self.rewems: np.ndarray | None = None
        self.gamma = gamma

    def update(self, rews: np.ndarray) -> np.ndarray:
        """Fold one step of per-env rewards into the running return."""
        self.rewems = rews if self.rewems is None else self.rewems * self.gamma + rews
        return self.rewems


class RNDModel(nn.Module):
    """Frozen random target network and trained predictor network.

    Both consume a single normalized frame of shape ``(1, 84, 84)``.
    """

    def __init__(self, feature_dim: int = 512) -> None:
        """Build target and predictor; freeze the target."""
        super().__init__()
        self.target = nn.Sequential(
            layer_init(nn.Conv2d(1, 32, 8, stride=4)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.LeakyReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, feature_dim)),
        )
        self.predictor = nn.Sequential(
            layer_init(nn.Conv2d(1, 32, 8, stride=4)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.LeakyReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, feature_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(feature_dim, feature_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(feature_dim, feature_dim)),
        )
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, next_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (predicted_feature, target_feature) for normalized frames."""
        return self.predictor(next_obs), self.target(next_obs)


def normalize_frames(
    frames: np.ndarray, rms: RunningMeanStd, device: torch.device
) -> torch.Tensor:
    """Normalize raw single frames with running stats and clip to [-5, 5].

    Args:
        frames: Raw uint8 frames, shape ``(N, 1, 84, 84)``.
        rms: Running mean/std over frames.
        device: Target device for the result.

    Returns:
        Float tensor of normalized, clipped frames.
    """
    arr = (frames.astype(np.float64) - rms.mean) / np.sqrt(rms.var)
    return torch.from_numpy(arr.clip(-5, 5).astype(np.float32)).to(device)
