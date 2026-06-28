"""Tests for the shared GAE implementation.

The headline result is produced by policies trained with this advantage math, so it is
pinned here: an equivalence test against the original inlined loop (the oracle) plus the
two analytic limits (lambda=1 Monte-Carlo, lambda=0 one-step TD) and done-masking.
"""

import torch

from dkrl.gae import compute_gae


def _reference_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    next_done: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The original inlined GAE loop, kept verbatim as an oracle for the extraction."""
    num_steps, num_envs = rewards.shape[0], rewards.shape[1]
    advantages = torch.zeros_like(rewards)
    lastgaelam = torch.zeros(num_envs)
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            nextnonterminal = 1.0 - next_done
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t + 1]
            nextvalues = values[t + 1]
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
        advantages[t] = lastgaelam
    return advantages, advantages + values


def test_matches_original_inlined_loop() -> None:
    """compute_gae reproduces the old inlined loop bit-for-bit, including next_value [1, N]."""
    torch.manual_seed(0)
    num_steps, num_envs = 7, 4
    rewards = torch.randn(num_steps, num_envs)
    values = torch.randn(num_steps, num_envs)
    dones = (torch.rand(num_steps, num_envs) < 0.2).float()
    next_value = torch.randn(1, num_envs)  # mirrors the .reshape(1, -1) call sites
    next_done = (torch.rand(num_envs) < 0.2).float()
    adv, ret = compute_gae(rewards, values, dones, next_value, next_done, 0.99, 0.95)
    ref_adv, ref_ret = _reference_gae(rewards, values, dones, next_value, next_done, 0.99, 0.95)
    assert torch.allclose(adv, ref_adv)
    assert torch.allclose(ret, ref_ret)


def test_lambda_one_is_monte_carlo_returns() -> None:
    """lambda=1 with zero baselines gives discounted reward-to-go bootstrapped on next_value."""
    rewards = torch.ones(4, 1)
    values = torch.zeros(4, 1)
    dones = torch.zeros(4, 1)
    next_value = torch.zeros(1, 1)
    next_done = torch.zeros(1)
    _, returns = compute_gae(rewards, values, dones, next_value, next_done, 0.5, 1.0)
    assert torch.allclose(returns, torch.tensor([[1.875], [1.75], [1.5], [1.0]]))


def test_lambda_zero_is_one_step_td() -> None:
    """lambda=0 gives advantage_t = r_t + gamma * V(s_{t+1}) - V(s_t)."""
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    values = torch.tensor([[0.5], [0.5], [0.5]])
    dones = torch.zeros(3, 1)
    next_value = torch.zeros(1, 1)
    next_done = torch.zeros(1)
    adv, _ = compute_gae(rewards, values, dones, next_value, next_done, 0.9, 0.0)
    expected = torch.tensor([[1.0 + 0.9 * 0.5 - 0.5], [2.0 + 0.9 * 0.5 - 0.5], [3.0 - 0.5]])
    assert torch.allclose(adv, expected)


def test_done_masks_the_bootstrap() -> None:
    """A done at t+1 zeroes the bootstrap from step t into step t+1."""
    rewards = torch.tensor([[1.0], [1.0]])
    values = torch.zeros(2, 1)
    dones = torch.tensor([[0.0], [1.0]])  # the episode ended at step 1
    next_value = torch.tensor([[5.0]])
    next_done = torch.zeros(1)
    adv, _ = compute_gae(rewards, values, dones, next_value, next_done, 0.99, 0.95)
    # Last step bootstraps next_value; step 0's bootstrap into step 1 is masked by dones[1]=1.
    assert torch.allclose(adv[1], torch.tensor([1.0 + 0.99 * 5.0]))
    assert torch.allclose(adv[0], torch.tensor([1.0]))
