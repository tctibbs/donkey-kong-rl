"""PPO actor-critic with separate extrinsic and intrinsic value heads."""

import torch
from torch import nn
from torch.distributions import Categorical

from dkrl.agents.ppo import layer_init


class PPORNDAgent(nn.Module):
    """Convolutional actor-critic with dual value heads for RND training."""

    def __init__(self, n_actions: int, frame_stack: int = 4) -> None:
        """Build the shared trunk, actor, and both critics.

        Args:
            n_actions: Size of the discrete action space.
            frame_stack: Number of stacked input frames (input channels).
        """
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(frame_stack, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, n_actions), std=0.01)
        self.critic_ext = layer_init(nn.Linear(512, 1), std=0.01)
        self.critic_int = layer_init(nn.Linear(512, 1), std=0.01)

    def get_value(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (extrinsic, intrinsic) value estimates."""
        hidden = self.network(x / 255.0)
        return self.critic_ext(hidden), self.critic_int(hidden)

    def get_action_and_value(
        self, x: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample (or score) actions and return both value estimates.

        Args:
            x: Batch of uint8 observations, shape ``(N, C, H, W)``.
            action: If given, score these actions instead of sampling.

        Returns:
            Tuple of ``(action, log_prob, entropy, value_ext, value_int)``.
        """
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return (
            action,
            dist.log_prob(action),
            dist.entropy(),
            self.critic_ext(hidden),
            self.critic_int(hidden),
        )
