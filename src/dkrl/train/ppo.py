"""PPO training loop on EnvPool, adapted from CleanRL ppo_atari_envpool."""

import csv
import random
import time
from collections import deque
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from dkrl.agents.ppo import PPOAgent
from dkrl.config import ExperimentConfig
from dkrl.envs import EpisodeStats, make_train_envs
from dkrl.ledger import git_sha
from dkrl.log import setup_logging


def make_run_id(cfg: ExperimentConfig) -> str:
    """Build a unique, sortable run id from name, seed, and UTC time."""
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return f"{cfg.name}_s{cfg.seed}_{stamp}"


def train(cfg: ExperimentConfig) -> dict:
    """Run PPO training to completion and return a ledger row.

    Args:
        cfg: Full experiment configuration.

    Returns:
        Ledger row dict with run identity, provenance, and training metrics.
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

    envs = make_train_envs(cfg.env, cfg.seed)
    n_actions = int(envs.action_space.n)
    agent = PPOAgent(n_actions, cfg.env.frame_stack).to(device)
    bc_stats: dict = {}
    if cfg.bc.enabled:
        from dkrl.bc import pretrain_bc

        bc_stats = pretrain_bc(agent, cfg.bc, device)
    demo_obs = demo_actions = None
    if cfg.bc.enabled and cfg.bc.aux_coef > 0:
        demo_data = np.load(cfg.bc.demo_path)
        demo_obs = torch.from_numpy(demo_data["observations"])
        demo_actions = torch.from_numpy(demo_data["actions"])
        logger.info(
            f"bc auxiliary loss active: coef={cfg.bc.aux_coef} over "
            f"{cfg.bc.aux_anneal_updates} updates, {len(demo_actions)} demo samples"
        )
    optimizer = optim.Adam(agent.parameters(), lr=cfg.ppo.learning_rate, eps=1e-5)

    num_envs = cfg.env.num_envs
    num_steps = cfg.ppo.num_steps
    batch_size = num_envs * num_steps
    minibatch_size = batch_size // cfg.ppo.num_minibatches
    num_updates = cfg.total_timesteps // batch_size

    obs_shape = (cfg.env.frame_stack, cfg.env.img_size, cfg.env.img_size)
    obs = torch.zeros((num_steps, num_envs, *obs_shape), dtype=torch.uint8, device=device)
    actions = torch.zeros((num_steps, num_envs), dtype=torch.long, device=device)
    logprobs = torch.zeros((num_steps, num_envs), device=device)
    rewards = torch.zeros((num_steps, num_envs), device=device)
    dones = torch.zeros((num_steps, num_envs), device=device)
    values = torch.zeros((num_steps, num_envs), device=device)

    stats = EpisodeStats(num_envs)
    recent_returns: deque[float] = deque(maxlen=100)
    recent_clears: deque[float] = deque(maxlen=100)
    best_return_window = float("-inf")
    episodes_done = 0
    clears_done = 0
    clear_threshold = cfg.eval.clear_reward_threshold

    metrics_dir = Path("results/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"{run_id}.csv"
    metrics_file = metrics_path.open("w", newline="")
    metrics = csv.writer(metrics_file)
    metrics.writerow(
        ["global_step", "time_s", "mean_return_100", "episodes", "sps", "clear_rate_100"]
    )

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
    next_obs = torch.from_numpy(next_obs_np).to(device)
    next_done = torch.zeros(num_envs, device=device)

    for update in range(1, num_updates + 1):
        if cfg.ppo.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            optimizer.param_groups[0]["lr"] = frac * cfg.ppo.learning_rate

        for step in range(num_steps):
            global_step += num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs_np, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            if cfg.env.reward_transform == "scale":
                reward = reward * cfg.env.reward_scale
            rewards[step] = torch.from_numpy(reward).to(device)
            done_np = np.logical_or(terminated, truncated)
            next_obs = torch.from_numpy(next_obs_np).to(device)
            next_done = torch.from_numpy(done_np.astype(np.float32)).to(device)

            for ep_return, ep_length, ep_max in stats.update(terminated, truncated, info):
                episodes_done += 1
                cleared = float(ep_max >= clear_threshold)
                clears_done += int(cleared)
                recent_returns.append(ep_return)
                recent_clears.append(cleared)
                writer.add_scalar("charts/episodic_return", ep_return, global_step)
                writer.add_scalar("charts/episodic_length", ep_length, global_step)
                writer.add_scalar("charts/episode_cleared", cleared, global_step)

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
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

        b_obs = obs.reshape((-1, *obs_shape))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(batch_size)
        clipfracs = []
        for _ in range(cfg.ppo.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                mb_inds = b_inds[start : start + minibatch_size]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > cfg.ppo.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if cfg.ppo.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - cfg.ppo.clip_coef, 1 + cfg.ppo.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if cfg.ppo.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -cfg.ppo.clip_coef, cfg.ppo.clip_coef
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - cfg.ppo.ent_coef * entropy_loss + v_loss * cfg.ppo.vf_coef
                if demo_obs is not None and demo_actions is not None:
                    frac_left = max(0.0, 1.0 - (update - 1) / cfg.bc.aux_anneal_updates)
                    aux_now = (
                        cfg.bc.aux_coef_floor
                        + (cfg.bc.aux_coef - cfg.bc.aux_coef_floor) * frac_left
                    )
                    if aux_now > 0:
                        idx = torch.randint(0, len(demo_actions), (cfg.bc.batch_size,))
                        d_obs = demo_obs[idx].to(device)
                        d_act = demo_actions[idx].to(device)
                        d_logits = agent.actor(agent.network(d_obs.float() / 255.0))
                        bc_aux_loss = torch.nn.functional.cross_entropy(d_logits, d_act)
                        loss = loss + aux_now * bc_aux_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), cfg.ppo.max_grad_norm)
                optimizer.step()

            if cfg.ppo.target_kl is not None and approx_kl > cfg.ppo.target_kl:
                break

        elapsed = time.time() - start_time
        sps = int(global_step / elapsed)
        mean_return = float(np.mean(recent_returns)) if recent_returns else float("nan")
        clear_rate_100 = float(np.mean(recent_clears)) if recent_clears else float("nan")
        if recent_returns and mean_return > best_return_window:
            best_return_window = mean_return

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", float(np.mean(clipfracs)), global_step)
        writer.add_scalar("charts/SPS", sps, global_step)
        writer.add_scalar("charts/clear_rate_100", clear_rate_100, global_step)
        metrics.writerow(
            [global_step, round(elapsed, 1), mean_return, episodes_done, sps, clear_rate_100]
        )
        metrics_file.flush()

        if update % 10 == 0 or update == num_updates:
            logger.info(
                f"update {update}/{num_updates} step={global_step} "
                f"return100={mean_return:.1f} clear100={clear_rate_100:.2f} "
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
        "total_frames": cfg.total_timesteps * cfg.env.frame_skip,
        "wall_clock_s": round(wall_clock_s, 1),
        "sps": int(global_step / wall_clock_s),
        "episodes": episodes_done,
        "train_clears": clears_done,
        "train_clear_rate_100_final": clear_rate_100,
        "train_return_100_final": mean_return,
        "train_return_100_best": best_return_window,
        "checkpoint": str(ckpt_dir / "final.pt"),
        "config": cfg.model_dump(mode="json"),
        "kind": "train",
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
        **bc_stats,
    }
    if cfg.track:
        import wandb

        row["wandb_url"] = wandb.run.get_url() if wandb.run is not None else None
        wandb.finish()
    logger.info(f"training done: best_return100={best_return_window:.1f} wall={wall_clock_s:.0f}s")
    return row
