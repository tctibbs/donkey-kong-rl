"""PPO with Random Network Distillation on EnvPool, after CleanRL ppo_rnd_envpool.

Differences from the plain PPO loop: a dual-head agent (extrinsic and
intrinsic critics), an RND predictor trained against a frozen random target,
non-episodic intrinsic returns with their own discount, and running
normalization for both RND observations and intrinsic rewards.
"""

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

from dkrl.agents.ppo_rnd import PPORNDAgent
from dkrl.config import ExperimentConfig
from dkrl.envs import EpisodeStats, make_train_envs
from dkrl.gae import compute_gae
from dkrl.ledger import git_sha
from dkrl.log import setup_logging
from dkrl.rnd import RewardForwardFilter, RNDModel, RunningMeanStd, normalize_frames
from dkrl.train.ppo import make_run_id


def train_rnd(cfg: ExperimentConfig) -> dict:
    """Run PPO+RND training to completion and return a ledger row.

    Args:
        cfg: Full experiment configuration with ``algo`` set to ``ppo_rnd``.

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
    agent = PPORNDAgent(n_actions, cfg.env.frame_stack).to(device)
    rnd = RNDModel(cfg.rnd.feature_dim).to(device)
    optimizer = optim.Adam(
        list(agent.parameters()) + list(rnd.predictor.parameters()),
        lr=cfg.ppo.learning_rate,
        eps=1e-5,
    )

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
    curiosity_rewards = torch.zeros((num_steps, num_envs), device=device)
    dones = torch.zeros((num_steps, num_envs), device=device)
    ext_values = torch.zeros((num_steps, num_envs), device=device)
    int_values = torch.zeros((num_steps, num_envs), device=device)
    rnd_frames = np.zeros(
        (num_steps, num_envs, 1, cfg.env.img_size, cfg.env.img_size), dtype=np.uint8
    )

    obs_rms = RunningMeanStd(shape=(1, cfg.env.img_size, cfg.env.img_size))
    reward_rms = RunningMeanStd()
    rff = RewardForwardFilter(cfg.rnd.int_gamma)

    stats = EpisodeStats(num_envs)
    recent_returns: deque[float] = deque(maxlen=100)
    recent_clears: deque[float] = deque(maxlen=100)
    best_return_window = float("-inf")
    episodes_done = 0
    clears_done = 0
    clear_threshold = cfg.eval.clear_reward_threshold

    metrics_dir = Path("results/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = (metrics_dir / f"{run_id}.csv").open("w", newline="")
    metrics = csv.writer(metrics_file)
    metrics.writerow(
        [
            "global_step",
            "time_s",
            "mean_return_100",
            "episodes",
            "sps",
            "clear_rate_100",
            "int_reward_mean",
        ]
    )

    ckpt_dir = Path("checkpoints") / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(name: str) -> None:
        torch.save(
            {
                "model_state_dict": agent.state_dict(),
                "rnd_state_dict": rnd.state_dict(),
                "n_actions": n_actions,
                "config": cfg.model_dump(mode="json"),
                "config_hash": cfg.config_hash,
                "git_sha": sha,
                "run_id": run_id,
            },
            ckpt_dir / f"{name}.pt",
        )

    logger.info(f"warming up obs normalization: {cfg.rnd.obs_norm_init_rollouts} rollouts")
    next_obs_np, _ = envs.reset()
    rng = np.random.default_rng(cfg.seed)
    for _ in range(cfg.rnd.obs_norm_init_rollouts):
        frames = []
        for _ in range(num_steps):
            acts = rng.integers(0, n_actions, size=num_envs)
            next_obs_np, _, _, _, _ = envs.step(acts)
            frames.append(next_obs_np[:, -1:, :, :])
        obs_rms.update(np.concatenate(frames).astype(np.float64))
    next_obs_np, _ = envs.reset()

    global_step = 0
    start_time = time.time()
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
                action, logprob, _, value_ext, value_int = agent.get_action_and_value(next_obs)
                ext_values[step] = value_ext.flatten()
                int_values[step] = value_int.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs_np, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            if cfg.env.reward_transform == "scale":
                reward = reward * cfg.env.reward_scale
            rewards[step] = torch.from_numpy(reward).to(device)
            done_np = np.logical_or(terminated, truncated)
            rnd_frames[step] = next_obs_np[:, -1:, :, :]
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
            flat_frames = rnd_frames.reshape((-1, 1, cfg.env.img_size, cfg.env.img_size))
            norm = normalize_frames(flat_frames, obs_rms, device)
            pred, target = rnd(norm)
            curiosity = ((pred - target) ** 2).sum(1) / 2
            curiosity_rewards[:] = curiosity.reshape(num_steps, num_envs)
        obs_rms.update(flat_frames.astype(np.float64))

        cur_np = curiosity_rewards.cpu().numpy()
        per_step_returns = np.array([rff.update(cur_np[i]) for i in range(num_steps)])
        reward_rms.update(per_step_returns.reshape(-1))
        curiosity_rewards /= float(np.sqrt(reward_rms.var)) + 1e-8

        with torch.no_grad():
            next_value_ext, next_value_int = agent.get_value(next_obs)
            next_value_ext = next_value_ext.reshape(1, -1)
            next_value_int = next_value_int.reshape(1, -1)

            ext_advantages, ext_returns = compute_gae(
                rewards,
                ext_values,
                dones,
                next_value_ext,
                next_done,
                cfg.ppo.gamma,
                cfg.ppo.gae_lambda,
            )
            int_advantages = torch.zeros_like(curiosity_rewards)
            int_lastgaelam = torch.zeros(num_envs, device=device)
            for t in reversed(range(num_steps)):
                int_nextvalues = next_value_int if t == num_steps - 1 else int_values[t + 1]
                int_delta = (
                    curiosity_rewards[t] + cfg.rnd.int_gamma * int_nextvalues - int_values[t]
                )
                int_lastgaelam = (
                    int_delta + cfg.rnd.int_gamma * cfg.ppo.gae_lambda * int_lastgaelam
                )
                int_advantages[t] = int_lastgaelam
            int_returns = int_advantages + int_values

        b_obs = obs.reshape((-1, *obs_shape))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_ext_advantages = ext_advantages.reshape(-1)
        b_int_advantages = int_advantages.reshape(-1)
        b_ext_returns = ext_returns.reshape(-1)
        b_int_returns = int_returns.reshape(-1)
        b_advantages = b_int_advantages * cfg.rnd.int_coef + b_ext_advantages * cfg.rnd.ext_coef
        b_frames = flat_frames

        b_inds = np.arange(batch_size)
        for _ in range(cfg.ppo.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                mb_inds = b_inds[start : start + minibatch_size]

                norm_mb = normalize_frames(b_frames[mb_inds], obs_rms, device)
                pred, target = rnd(norm_mb)
                forward_loss = ((pred - target.detach()) ** 2).mean(-1)
                mask = (
                    torch.rand(len(mb_inds), device=device) < cfg.rnd.update_proportion
                ).float()
                forward_loss = (forward_loss * mask).sum() / torch.max(
                    mask.sum(), torch.tensor(1.0, device=device)
                )

                _, newlogprob, entropy, new_ext_value, new_int_value = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()

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

                ext_v_loss = 0.5 * ((new_ext_value.view(-1) - b_ext_returns[mb_inds]) ** 2).mean()
                int_v_loss = 0.5 * ((new_int_value.view(-1) - b_int_returns[mb_inds]) ** 2).mean()
                v_loss = ext_v_loss + int_v_loss
                entropy_loss = entropy.mean()
                loss = (
                    pg_loss
                    - cfg.ppo.ent_coef * entropy_loss
                    + v_loss * cfg.ppo.vf_coef
                    + forward_loss
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(agent.parameters()) + list(rnd.predictor.parameters()),
                    cfg.ppo.max_grad_norm,
                )
                optimizer.step()

            if cfg.ppo.target_kl is not None and approx_kl > cfg.ppo.target_kl:
                break

        elapsed = time.time() - start_time
        sps = int(global_step / elapsed)
        mean_return = float(np.mean(recent_returns)) if recent_returns else float("nan")
        clear_rate_100 = float(np.mean(recent_clears)) if recent_clears else float("nan")
        int_reward_mean = float(curiosity_rewards.mean().item())
        if recent_returns and mean_return > best_return_window:
            best_return_window = mean_return

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/clear_rate_100", clear_rate_100, global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/forward_loss", forward_loss.item(), global_step)
        writer.add_scalar("charts/int_reward_mean", int_reward_mean, global_step)
        writer.add_scalar("charts/SPS", sps, global_step)
        metrics.writerow(
            [
                global_step,
                round(elapsed, 1),
                mean_return,
                episodes_done,
                sps,
                clear_rate_100,
                int_reward_mean,
            ]
        )
        metrics_file.flush()

        if update % 10 == 0 or update == num_updates:
            logger.info(
                f"update {update}/{num_updates} step={global_step} "
                f"return100={mean_return:.1f} clear100={clear_rate_100:.2f} "
                f"int_r={int_reward_mean:.3f} episodes={episodes_done} sps={sps}"
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
    }
    if cfg.track:
        import wandb

        row["wandb_url"] = wandb.run.get_url() if wandb.run is not None else None
        wandb.finish()
    logger.info(
        f"training done: best_return100={best_return_window:.1f} "
        f"clears={clears_done} wall={wall_clock_s:.0f}s"
    )
    return row
