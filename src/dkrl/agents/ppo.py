"""PPO actor-critic with the Nature CNN trunk, after CleanRL ppo_atari_envpool."""

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


def layer_init(
    layer: nn.Module, std: float = float(np.sqrt(2)), bias_const: float = 0.0
) -> nn.Module:
    """Orthogonally initialize a layer's weight and constant-fill its bias.

    Args:
        layer: Linear or convolutional layer to initialize.
        std: Gain for orthogonal initialization.
        bias_const: Constant for the bias.

    Returns:
        The same layer, initialized in place.
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PPOAgent(nn.Module):
    """Convolutional actor-critic for pixel Atari observations."""

    def __init__(self, n_actions: int, frame_stack: int = 4) -> None:
        """Build the network.

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
        self.critic = layer_init(nn.Linear(512, 1), std=1.0)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Return the value estimate for a batch of observations."""
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(
        self, x: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample (or score) actions and return value estimates.

        Args:
            x: Batch of uint8 observations, shape ``(N, C, H, W)``.
            action: If given, score these actions instead of sampling.

        Returns:
            Tuple of ``(action, log_prob, entropy, value)``.
        """
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(hidden)
