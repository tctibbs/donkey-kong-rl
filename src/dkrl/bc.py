"""Behavior-cloning warm-start from scripted demonstrations (ADR 0004)."""

from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch import nn, optim

from dkrl.agents.ppo import PPOAgent
from dkrl.config import BCConfig


def pretrain_bc(agent: PPOAgent, cfg: BCConfig, device: torch.device) -> dict:
    """Train the agent's policy head (and trunk) on demonstration pairs.

    Cross-entropy on actor logits over the demo dataset. The critic is left
    untouched; PPO fits values during fine-tuning.

    Args:
        agent: The PPO agent to warm-start, already on ``device``.
        cfg: Behavior-cloning configuration.
        device: Training device.

    Returns:
        Dict with final loss, accuracy, sample count, and epochs.
    """
    data = np.load(Path(cfg.demo_path))
    observations = torch.from_numpy(data["observations"])
    actions = torch.from_numpy(data["actions"])
    n = len(actions)
    logger.info(f"bc warm-start: {n} samples from {cfg.demo_path}")

    optimizer = optim.Adam(agent.parameters(), lr=cfg.learning_rate)
    ce = nn.CrossEntropyLoss()
    loss_val = float("nan")
    accuracy = float("nan")
    for epoch in range(cfg.epochs):
        perm = torch.randperm(n)
        correct = 0
        total_loss = 0.0
        batches = 0
        for start in range(0, n, cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            obs = observations[idx].to(device)
            act = actions[idx].to(device)
            logits = agent.actor(agent.network(obs.float() / 255.0))
            loss = ce(logits, act)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batches += 1
            correct += int((logits.argmax(dim=1) == act).sum().item())
        loss_val = total_loss / batches
        accuracy = correct / n
        logger.info(f"bc epoch {epoch + 1}/{cfg.epochs}: loss={loss_val:.4f} acc={accuracy:.3f}")
    return {"bc_loss": loss_val, "bc_accuracy": accuracy, "bc_samples": n, "bc_epochs": cfg.epochs}
