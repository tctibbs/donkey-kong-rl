"""Recurrent actor-critic: Nature CNN trunk feeding an LSTM, after CleanRL.

The LSTM lets the policy integrate enemy motion across time (flame velocity
and trajectory), which a feedforward policy over stacked frames cannot do.
Hidden state is carried across rollout steps and reset at episode boundaries.
"""

import torch
from torch import nn
from torch.distributions import Categorical

from dkrl.agents.ppo import layer_init


class LSTMAgent(nn.Module):
    """Convolutional-recurrent actor-critic for pixel Atari observations."""

    def __init__(self, n_actions: int, frame_stack: int = 4, lstm_hidden: int = 128) -> None:
        """Build the CNN trunk, the LSTM, and the actor and critic heads.

        Args:
            n_actions: Size of the discrete action space.
            frame_stack: Number of stacked input frames (input channels).
            lstm_hidden: LSTM hidden and cell size.
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
        self.lstm = nn.LSTM(512, lstm_hidden)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        self.actor = layer_init(nn.Linear(lstm_hidden, n_actions), std=0.01)
        self.critic = layer_init(nn.Linear(lstm_hidden, 1), std=1.0)
        self.lstm_hidden = lstm_hidden

    def initial_state(self, batch: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Return zeroed (h, c) hidden state for ``batch`` environments."""
        shape = (self.lstm.num_layers, batch, self.lstm_hidden)
        return (
            torch.zeros(shape, device=device),
            torch.zeros(shape, device=device),
        )

    def _lstm_forward(
        self,
        hidden: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
        done: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Run the LSTM over a flattened (T*N, feat) batch, resetting on done.

        Args:
            hidden: CNN features, shape ``(T*N, 512)``.
            state: Incoming (h, c) for N envs.
            done: Per-step done flags, shape ``(T*N,)``.

        Returns:
            Tuple of (lstm_outputs ``(T*N, hidden)``, final state).
        """
        n = state[0].shape[1]
        hidden = hidden.reshape(-1, n, self.lstm.input_size)
        done = done.reshape(-1, n)
        outputs = []
        for h_t, d_t in zip(hidden, done, strict=True):
            keep = (1.0 - d_t).view(1, -1, 1)
            state = (state[0] * keep, state[1] * keep)
            out, state = self.lstm(h_t.unsqueeze(0), state)
            outputs.append(out)
        return torch.flatten(torch.cat(outputs), 0, 1), state

    def get_states(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
        done: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Encode observations through CNN and LSTM."""
        hidden = self.network(x / 255.0)
        return self._lstm_forward(hidden, state, done)

    def get_value(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
        done: torch.Tensor,
    ) -> torch.Tensor:
        """Return value estimates for a batch."""
        out, _ = self.get_states(x, state, done)
        return self.critic(out)

    def get_action_and_value(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
        done: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple]:
        """Sample (or score) actions and return values plus the new state.

        Args:
            x: uint8 observations ``(T*N, C, H, W)``.
            state: Incoming (h, c).
            done: Per-step done flags ``(T*N,)``.
            action: If given, score these actions instead of sampling.

        Returns:
            Tuple of ``(action, log_prob, entropy, value, new_state)``.
        """
        out, new_state = self.get_states(x, state, done)
        logits = self.actor(out)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(out), new_state
