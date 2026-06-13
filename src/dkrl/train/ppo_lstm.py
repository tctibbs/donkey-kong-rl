"""Recurrent PPO on EnvPool, after CleanRL ppo_atari_lstm.

Carries LSTM hidden state across the rollout, resets it at episode
boundaries, and trains on whole-env sequences (minibatches are over envs, not
flattened timesteps) so the recurrence is preserved. Supports BC warm-start
plus the annealed BC auxiliary floor, identical in spirit to train/ppo.py.
"""

import csv
import random
import time
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from dkrl.agents.lstm import LSTMAgent
from dkrl.config import ExperimentConfig
from dkrl.envs import EpisodeStats, make_train_envs
from dkrl.ledger import git_sha
from dkrl.log import setup_logging
from dkrl.train.ppo import make_run_id


def _distill_teacher(
    agent: LSTMAgent,
    cfg: ExperimentConfig,
    device: torch.device,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Distill a feedforward teacher into the LSTM with correct recurrence.

    Rolls the teacher policy through the training env in (num_steps, num_envs)
    chunks (same shape the PPO update uses), then trains the LSTM to match the
    teacher's actions over each chunk with hidden-state propagation and
    done-based resets. This gives the LSTM both the teacher's visual features
    and an action prior that is valid under real recurrent rollout, unlike
    single-frame BC. The teacher callable is returned so the PPO phase can keep
    an annealed online-distillation loss (the recurrent equivalent of the BC
    floor) to hold the behavior while fine-tuning.
    """
    from dkrl.evaluate import load_policy

    assert cfg.lstm.distill_teacher is not None
    teacher_ckpt = torch.load(cfg.lstm.distill_teacher, map_location="cpu", weights_only=False)
    teacher_cfg = ExperimentConfig.model_validate(teacher_ckpt["config"])
    teacher = load_policy(teacher_ckpt, teacher_cfg, device)

    # Warm-start the CNN trunk from the teacher (identical architecture).
    teacher_net = {
        k[len("network.") :]: v
        for k, v in teacher_ckpt["model_state_dict"].items()
        if k.startswith("network.")
    }
    agent.network.load_state_dict(teacher_net)

    envs = make_train_envs(cfg.env, cfg.seed + 7)
    num_envs = cfg.env.num_envs
    num_steps = cfg.ppo.num_steps
    obs_shape = (cfg.env.frame_stack, cfg.env.img_size, cfg.env.img_size)
    opt = optim.Adam(agent.parameters(), lr=cfg.bc.learning_rate)
    ce = torch.nn.CrossEntropyLoss()

    next_obs_np, _ = envs.reset()
    next_obs = torch.from_numpy(next_obs_np).to(device)
    next_done = torch.zeros(num_envs, device=device)
    for chunk in range(cfg.lstm.distill_chunks):
        chunk_obs = torch.zeros(
            (num_steps, num_envs, *obs_shape), dtype=torch.uint8, device=device
        )
        chunk_act = torch.zeros((num_steps, num_envs), dtype=torch.long, device=device)
        chunk_done = torch.zeros((num_steps, num_envs), device=device)
        for step in range(num_steps):
            chunk_obs[step] = next_obs
            chunk_done[step] = next_done
            with torch.no_grad():
                teacher_action = teacher(next_obs)
            chunk_act[step] = teacher_action
            next_obs_np, _, terminated, truncated, _ = envs.step(teacher_action.cpu().numpy())
            next_obs = torch.from_numpy(next_obs_np).to(device)
            next_done = torch.from_numpy(
                np.logical_or(terminated, truncated).astype(np.float32)
            ).to(device)

        state = agent.initial_state(num_envs, device)
        out, _ = agent.get_states(
            chunk_obs.reshape((-1, *obs_shape)).float(), state, chunk_done.reshape(-1)
        )
        logits = agent.actor(out)
        loss = ce(logits, chunk_act.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), cfg.ppo.max_grad_norm)
        opt.step()
        if (chunk + 1) % 20 == 0:
            acc = float((logits.argmax(1) == chunk_act.reshape(-1)).float().mean())
            logger.info(f"distill chunk {chunk + 1}/{cfg.lstm.distill_chunks}: acc={acc:.3f}")
    envs.close()
    return teacher


def _lstm_bc_logits(agent: LSTMAgent, frames: torch.Tensor) -> torch.Tensor:
    """Actor logits for a batch of single frames (zeroed hidden, length-1 seq)."""
    device = frames.device
    state = agent.initial_state(frames.shape[0], device)
    done = torch.zeros(frames.shape[0], device=device)
    out, _ = agent.get_states(frames.float(), state, done)
    return agent.actor(out)


def _lstm_bc_pretrain(
    agent: LSTMAgent,
    demo_obs: torch.Tensor,
    demo_actions: torch.Tensor,
    cfg: ExperimentConfig,
    device: torch.device,
) -> None:
    """Warm-start the LSTM agent on demo frames (each as a length-1 sequence)."""
    opt = optim.Adam(agent.parameters(), lr=cfg.bc.learning_rate)
    ce = torch.nn.CrossEntropyLoss()
    n = len(demo_actions)
    for epoch in range(cfg.bc.epochs):
        perm = torch.randperm(n)
        correct = 0
        for start in range(0, n, cfg.bc.batch_size):
            idx = perm[start : start + cfg.bc.batch_size]
            frames = demo_obs[idx].to(device)
            act = demo_actions[idx].to(device)
            logits = _lstm_bc_logits(agent, frames)
            loss = ce(logits, act)
            opt.zero_grad()
            loss.backward()
            opt.step()
            correct += int((logits.argmax(1) == act).sum().item())
        logger.info(f"lstm bc epoch {epoch + 1}/{cfg.bc.epochs}: acc={correct / n:.3f}")


def train_lstm(cfg: ExperimentConfig) -> dict:
    """Run recurrent PPO training to completion and return a ledger row.

    Args:
        cfg: Full experiment configuration with ``algo`` set to ``ppo_lstm``.

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

    envs = make_train_envs(cfg.env, cfg.seed)
    n_actions = int(envs.action_space.n)
    agent = LSTMAgent(n_actions, cfg.env.frame_stack, cfg.lstm.hidden_size).to(device)

    demo_obs = demo_actions = None
    teacher = None
    if cfg.lstm.distill_teacher:
        teacher = _distill_teacher(agent, cfg, device)
    elif cfg.bc.enabled:
        demo_data = np.load(cfg.bc.demo_path)
        demo_obs = torch.from_numpy(demo_data["observations"])
        demo_actions = torch.from_numpy(demo_data["actions"])
        _lstm_bc_pretrain(agent, demo_obs, demo_actions, cfg, device)
        if cfg.bc.aux_coef <= 0:
            demo_obs = demo_actions = None

    optimizer = optim.Adam(agent.parameters(), lr=cfg.ppo.learning_rate, eps=1e-5)

    num_envs = cfg.env.num_envs
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
    next_state = agent.initial_state(num_envs, device)

    for update in range(1, num_updates + 1):
        initial_state = (next_state[0].clone(), next_state[1].clone())
        if cfg.ppo.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            optimizer.param_groups[0]["lr"] = frac * cfg.ppo.learning_rate

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

            next_obs_np, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            if cfg.env.reward_transform == "scale":
                reward = reward * cfg.env.reward_scale
            rewards[step] = torch.from_numpy(reward).to(device)
            done_np = np.logical_or(terminated, truncated)
            next_obs = torch.from_numpy(next_obs_np).to(device)
            next_done = torch.from_numpy(done_np.astype(np.float32)).to(device)
            for ep_return, _ep_length, ep_max in stats.update(terminated, truncated, info):
                episodes_done += 1
                cleared = float(ep_max >= clear_threshold)
                clears_done += int(cleared)
                recent_returns.append(ep_return)
                recent_clears.append(cleared)
                writer.add_scalar("charts/episodic_return", ep_return, global_step)
                writer.add_scalar("charts/episode_cleared", cleared, global_step)

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
        flat_dones = dones.reshape(-1)
        for _ in range(cfg.ppo.update_epochs):
            np.random.shuffle(env_inds)
            for start in range(0, num_envs, envs_per_batch):
                mb_envs = env_inds[start : start + envs_per_batch]
                mb_obs = obs[:, mb_envs].reshape((-1, *obs_shape))
                mb_dones = dones[:, mb_envs].reshape(-1)
                mb_state = (initial_state[0][:, mb_envs], initial_state[1][:, mb_envs])
                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                    mb_obs, mb_state, mb_dones, actions[:, mb_envs].reshape(-1)
                )
                mb_logprobs = logprobs[:, mb_envs].reshape(-1)
                mb_adv = advantages[:, mb_envs].reshape(-1)
                mb_ret = returns[:, mb_envs].reshape(-1)
                logratio = newlogprob - mb_logprobs
                ratio = logratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                if cfg.ppo.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg_loss = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * torch.clamp(ratio, 1 - cfg.ppo.clip_coef, 1 + cfg.ppo.clip_coef),
                ).mean()
                v_loss = 0.5 * ((newvalue.view(-1) - mb_ret) ** 2).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - cfg.ppo.ent_coef * entropy_loss + v_loss * cfg.ppo.vf_coef
                if teacher is not None:
                    frac_left = max(0.0, 1.0 - (update - 1) / cfg.bc.aux_anneal_updates)
                    aux_now = (
                        cfg.bc.aux_coef_floor
                        + (cfg.bc.aux_coef - cfg.bc.aux_coef_floor) * frac_left
                    )
                    if aux_now > 0:
                        with torch.no_grad():
                            teacher_act = teacher(mb_obs)
                        mb_logits = agent.actor(
                            agent.get_states(
                                mb_obs.float(), mb_state, dones[:, mb_envs].reshape(-1)
                            )[0]
                        )
                        distill_loss = torch.nn.functional.cross_entropy(mb_logits, teacher_act)
                        loss = loss + aux_now * distill_loss
                if demo_obs is not None and demo_actions is not None:
                    frac_left = max(0.0, 1.0 - (update - 1) / cfg.bc.aux_anneal_updates)
                    aux_now = (
                        cfg.bc.aux_coef_floor
                        + (cfg.bc.aux_coef - cfg.bc.aux_coef_floor) * frac_left
                    )
                    if aux_now > 0:
                        d_idx = torch.randint(0, len(demo_actions), (cfg.bc.batch_size,))
                        d_logits = _lstm_bc_logits(agent, demo_obs[d_idx].to(device))
                        bc_aux = torch.nn.functional.cross_entropy(
                            d_logits, demo_actions[d_idx].to(device)
                        )
                        loss = loss + aux_now * bc_aux
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), cfg.ppo.max_grad_norm)
                optimizer.step()
            if cfg.ppo.target_kl is not None and approx_kl > cfg.ppo.target_kl:
                break
        del flat_dones

        elapsed = time.time() - start_time
        sps = int(global_step / elapsed)
        mean_return = float(np.mean(recent_returns)) if recent_returns else float("nan")
        clear_rate_100 = float(np.mean(recent_clears)) if recent_clears else float("nan")
        if recent_returns and mean_return > best_return_window:
            best_return_window = mean_return
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
    logger.info(f"lstm training done: best_return100={best_return_window:.1f}")
    return row
