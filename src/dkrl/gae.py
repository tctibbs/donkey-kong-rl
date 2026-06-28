"""Generalized Advantage Estimation.

Factored out of the training loops so every PPO variant shares one tested implementation
of the advantage and return computation.
"""

import torch


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    next_done: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE advantages and bootstrapped returns for one rollout.

    Args:
        rewards: Per-step rewards, shape ``(num_steps, num_envs)``.
        values: Critic value estimates, shape ``(num_steps, num_envs)``.
        dones: Per-step done flags (1.0 when the step ended an episode), shape
            ``(num_steps, num_envs)``.
        next_value: Bootstrap value of the state after the last step, broadcastable to
            ``(num_envs,)``.
        next_done: Done flag for the state after the last step, shape ``(num_envs,)``.
        gamma: Discount factor.
        gae_lambda: GAE lambda.

    Returns:
        ``(advantages, returns)``, each shape ``(num_steps, num_envs)``, with
        ``returns == advantages + values``.
    """
    num_steps, num_envs = rewards.shape[0], rewards.shape[1]
    advantages = torch.zeros_like(rewards)
    lastgaelam = torch.zeros(num_envs, device=rewards.device)
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            nextnonterminal = 1.0 - next_done
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t + 1]
            nextvalues = values[t + 1]
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        advantages[t] = lastgaelam
    returns = advantages + values
    return advantages, returns
