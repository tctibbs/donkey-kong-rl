"""Recurrent backward-curriculum robustification.

Combines the restore-reset curriculum env (Go-Explore phase 2) with the LSTM
policy: the best recurrent agent practices deep game states by trial under
real sticky noise, carrying hidden state so it can use flame-velocity memory
within each restored episode. This is the synthesis of the two levers that
each helped (recurrence and deep-state practice).
"""

import csv
import random
import time
from collections import deque
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from loguru import logger
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from dkrl.agents.lstm import LSTMAgent
from dkrl.config import ExperimentConfig
from dkrl.ledger import git_sha
from dkrl.log import setup_logging
from dkrl.robustify import RestoreResetEnv
from dkrl.train.ppo import make_run_id


def _worker(  # noqa: ANN202
    bank_path: str, board: int | None, sticky: float, max_steps: int, seed: int, start_prob: float
):
    """Factory for one restore-reset worker env."""
    return RestoreResetEnv(
        bank_path,
        board_filter=board,
        sticky=sticky,
        max_steps=max_steps,
        seed=seed,
        start_reset_prob=start_prob,
    )


def train_robustify_lstm(cfg: ExperimentConfig) -> dict:
    """Run recurrent backward-curriculum PPO and return a ledger row.

    Args:
        cfg: Config with algo ``ppo_robustify`` and robustify.recurrent true.

    Returns:
        Ledger row dict.
    """
    run_id = make_run_id(cfg)
    run_dir = Path("runs") / run_id
    setup_logging(run_id, run_dir)
    sha = git_sha()
    logger.info(f"config_hash={cfg.config_hash} git_sha={sha}")
    logger.info(f"config: {cfg.model_dump_json()}")

    if cfg.track:
        import wandb

        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=run_id,
            config={**cfg.model_dump(), "config_hash": cfg.config_hash, "git_sha": sha},
            sync_tensorboard=True,
        )
    writer = SummaryWriter(str(run_dir))

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.cuda else "cpu")
    logger.info(f"device={device}")

    rb = cfg.robustify
    num_envs = cfg.env.num_envs
    envs = gym.vector.AsyncVectorEnv(
        [
            partial(
                _worker,
                rb.bank_path,
                rb.board_filter,
                cfg.env.sticky_action_prob,
                rb.max_episode_steps,
                cfg.seed * 1000 + i,
                rb.start_reset_prob,
            )
            for i in range(num_envs)
        ],
        context="spawn",
    )
    action_space = envs.single_action_space
    assert isinstance(action_space, gym.spaces.Discrete)
    n_actions = int(action_space.n)
    agent = LSTMAgent(n_actions, cfg.env.frame_stack, cfg.lstm.hidden_size).to(device)
    if rb.init_from:
        init_ckpt = torch.load(rb.init_from, map_location="cpu", weights_only=False)
        agent.load_state_dict(init_ckpt["model_state_dict"])
        logger.info(f"warm-started recurrent policy from {rb.init_from}")
    optimizer = optim.Adam(agent.parameters(), lr=cfg.ppo.learning_rate, eps=1e-5)

    num_steps = cfg.ppo.num_steps
    batch_size = num_envs * num_steps
    num_updates = cfg.total_timesteps // batch_size
    envs_per_batch = num_envs // cfg.ppo.num_minibatches
    obs_shape = (cfg.env.frame_stack, cfg.env.img_size, cfg.env.img_size)
    obs = torch.zeros((num_steps, num_envs, *obs_shape), dtype=torch.uint8, device=device)
    actions = torch.zeros((num_steps, num_envs), dtype=torch.long, device=device)
    logprobs = torch.zeros((num_steps, num_envs), device=device)
    rewards = torch.zeros((num_steps, num_envs), device=device)
    dones = torch.zeros((num_steps, num_envs), device=device)
    values = torch.zeros((num_steps, num_envs), device=device)

    ep_returns_acc = np.zeros(num_envs, dtype=np.float64)
    recent_returns: deque[float] = deque(maxlen=100)
    best_return_window = float("-inf")
    episodes_done = 0

    metrics_dir = Path("results/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = (metrics_dir / f"{run_id}.csv").open("w", newline="")
    metrics = csv.writer(metrics_file)
    metrics.writerow(["global_step", "time_s", "mean_return_100", "episodes", "sps", "window_hi"])
    ckpt_dir = Path("checkpoints") / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(name: str) -> None:
        torch.save(
            {
                "model_state_dict": agent.state_dict(),
                "n_actions": n_actions,
                "config": cfg.model_dump(mode="json"),
                "config_hash": cfg.config_hash,
                "git_sha": sha,
                "run_id": run_id,
            },
            ckpt_dir / f"{name}.pt",
        )

    global_step = 0
    start_time = time.time()
    next_obs_np, _ = envs.reset()
    next_obs = torch.from_numpy(np.asarray(next_obs_np)).to(device)
    next_done = torch.zeros(num_envs, device=device)
    next_state = agent.initial_state(num_envs, device)
    window_hi = rb.start_window

    for update in range(1, num_updates + 1):
        initial_state = (next_state[0].clone(), next_state[1].clone())
        frac_done = (update - 1) / max(1, num_updates)
        window_hi = min(
            1.0, rb.start_window + (1.0 - rb.start_window) * (frac_done / rb.schedule_frac)
        )
        envs.call("set_window", 0.0, window_hi)
        if cfg.ppo.anneal_lr:
            optimizer.param_groups[0]["lr"] = (1.0 - frac_done) * cfg.ppo.learning_rate

        for step in range(num_steps):
            global_step += num_envs
            obs[step] = next_obs
            dones[step] = next_done
            with torch.no_grad():
                action, logprob, _, value, next_state = agent.get_action_and_value(
                    next_obs, next_state, next_done
                )
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob
            next_obs_np, reward, terminated, truncated, _ = envs.step(action.cpu().numpy())
            ep_returns_acc += reward
            done_np = np.logical_or(terminated, truncated)
            for i in np.nonzero(done_np)[0]:
                episodes_done += 1
                recent_returns.append(float(ep_returns_acc[i]))
                writer.add_scalar("charts/episodic_return", ep_returns_acc[i], global_step)
                ep_returns_acc[i] = 0.0
            if cfg.env.reward_transform == "scale":
                reward = reward * cfg.env.reward_scale
            rewards[step] = torch.from_numpy(reward.astype(np.float32)).to(device)
            next_obs = torch.from_numpy(np.asarray(next_obs_np)).to(device)
            next_done = torch.from_numpy(done_np.astype(np.float32)).to(device)

        with torch.no_grad():
            next_value = agent.get_value(next_obs, next_state, next_done).reshape(1, -1)
            advantages = torch.zeros_like(rewards)
            lastgaelam = torch.zeros(num_envs, device=device)
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + cfg.ppo.gamma * nextvalues * nextnonterminal - values[t]
                lastgaelam = (
                    delta + cfg.ppo.gamma * cfg.ppo.gae_lambda * nextnonterminal * lastgaelam
                )
                advantages[t] = lastgaelam
            returns = advantages + values

        env_inds = np.arange(num_envs)
        for _ in range(cfg.ppo.update_epochs):
            np.random.shuffle(env_inds)
            for start in range(0, num_envs, envs_per_batch):
                mb = env_inds[start : start + envs_per_batch]
                mb_state = (initial_state[0][:, mb], initial_state[1][:, mb])
                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                    obs[:, mb].reshape((-1, *obs_shape)),
                    mb_state,
                    dones[:, mb].reshape(-1),
                    actions[:, mb].reshape(-1),
                )
                logratio = newlogprob - logprobs[:, mb].reshape(-1)
                ratio = logratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                mb_adv = advantages[:, mb].reshape(-1)
                if cfg.ppo.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg_loss = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * torch.clamp(ratio, 1 - cfg.ppo.clip_coef, 1 + cfg.ppo.clip_coef),
                ).mean()
                v_loss = 0.5 * ((newvalue.view(-1) - returns[:, mb].reshape(-1)) ** 2).mean()
                loss = pg_loss - cfg.ppo.ent_coef * entropy.mean() + v_loss * cfg.ppo.vf_coef
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), cfg.ppo.max_grad_norm)
                optimizer.step()
            if cfg.ppo.target_kl is not None and approx_kl > cfg.ppo.target_kl:
                break

        elapsed = time.time() - start_time
        sps = int(global_step / elapsed)
        mean_return = float(np.mean(recent_returns)) if recent_returns else float("nan")
        if recent_returns and mean_return > best_return_window:
            best_return_window = mean_return
        writer.add_scalar("charts/window_hi", window_hi, global_step)
        writer.add_scalar("charts/SPS", sps, global_step)
        metrics.writerow(
            [global_step, round(elapsed, 1), mean_return, episodes_done, sps, round(window_hi, 3)]
        )
        metrics_file.flush()
        if update % 10 == 0 or update == num_updates:
            logger.info(
                f"update {update}/{num_updates} step={global_step} "
                f"return100={mean_return:.1f} window={window_hi:.2f} "
                f"episodes={episodes_done} sps={sps}"
            )
        if cfg.checkpoint_interval_updates and update % cfg.checkpoint_interval_updates == 0:
            save_checkpoint(f"update_{update:06d}")

    save_checkpoint("final")
    metrics_file.close()
    envs.close()
    writer.close()

    wall_clock_s = time.time() - start_time
    row = {
        "run_id": run_id,
        "name": cfg.name,
        "algo": cfg.algo,
        "config_hash": cfg.config_hash,
        "git_sha": sha,
        "seed": cfg.seed,
        "total_timesteps": cfg.total_timesteps,
        "wall_clock_s": round(wall_clock_s, 1),
        "sps": int(global_step / wall_clock_s),
        "episodes": episodes_done,
        "train_return_100_final": mean_return,
        "train_return_100_best": best_return_window,
        "checkpoint": str(ckpt_dir / "final.pt"),
        "config": cfg.model_dump(mode="json"),
        "kind": "train",
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    if cfg.track:
        import wandb

        row["wandb_url"] = wandb.run.get_url() if wandb.run is not None else None
        wandb.finish()
    logger.info(f"recurrent robustify done: best_return100={best_return_window:.1f}")
    return row
