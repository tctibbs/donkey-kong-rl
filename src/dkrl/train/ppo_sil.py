"""PPO with Self-Imitation Learning (Oh et al. 2018) on EnvPool.

SIL augments PPO with an off-policy update that replays the agent's own
high-return transitions: for stored (state, action, return R), it pushes the
policy toward the action in proportion to the positive advantage (R - V(s))+.
The replay buffer is SEEDED with deep Go-Explore trajectories, so the policy
self-imitates 30k-level play inside the on-policy loop, warm-started from the
best feedforward agent. This targets the measured failure (converting rare
high-return experience into a robust closed-loop policy) without behavior
cloning's covariate shift or restore-reset's distribution shift.
"""

import csv
import pickle
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
from dkrl.gae import compute_gae
from dkrl.ledger import git_sha
from dkrl.log import setup_logging
from dkrl.train.ppo import make_run_id


class SILBuffer:
    """Self-imitation buffer with a circular self-play region and a separate,
    NON-EVICTING demonstration region (SIL-from-Demonstrations design).

    Seeded demonstration transitions are kept permanently so the policy keeps
    self-imitating the deep trajectories instead of overwriting them with its
    own lower-return episodes. Sampling draws a fixed fraction from the demos.
    """

    def __init__(self, capacity: int, obs_shape: tuple[int, ...], demo_capacity: int) -> None:
        """Allocate the self-play and demonstration arrays."""
        self.capacity = capacity
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.returns = np.zeros(capacity, dtype=np.float32)
        self.size = 0
        self.pos = 0
        self.d_obs = np.zeros((demo_capacity, *obs_shape), dtype=np.uint8)
        self.d_actions = np.zeros(demo_capacity, dtype=np.int64)
        self.d_returns = np.zeros(demo_capacity, dtype=np.float32)
        self.d_size = 0
        self.d_capacity = demo_capacity

    def add_episode(self, obs: list[np.ndarray], actions: list[int], returns: list[float]) -> None:
        """Add one self-play episode's transitions to the circular region."""
        for o, a, r in zip(obs, actions, returns, strict=True):
            self.obs[self.pos] = o
            self.actions[self.pos] = a
            self.returns[self.pos] = r
            self.pos = (self.pos + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def add_demos(self, obs: list[np.ndarray], actions: list[int], returns: list[float]) -> None:
        """Add demonstration transitions to the permanent, non-evicting region."""
        for o, a, r in zip(obs, actions, returns, strict=True):
            if self.d_size >= self.d_capacity:
                break
            self.d_obs[self.d_size] = o
            self.d_actions[self.d_size] = a
            self.d_returns[self.d_size] = r
            self.d_size += 1

    @staticmethod
    def _prioritized(returns: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
        """Return k indices sampled proportional to (return - min + 1)."""
        priorities = returns - returns.min() + 1.0
        return rng.choice(len(returns), size=k, p=priorities / priorities.sum())

    def sample(
        self, batch: int, demo_fraction: float, rng: np.random.Generator
    ) -> tuple[np.ndarray, ...]:
        """Sample, drawing demo_fraction from the permanent demo region."""
        n_demo = int(batch * demo_fraction) if self.d_size > 0 else 0
        n_self = batch - n_demo
        obs_parts, act_parts, ret_parts = [], [], []
        if n_self > 0 and self.size > 0:
            i = self._prioritized(self.returns[: self.size], min(n_self, self.size), rng)
            obs_parts.append(self.obs[i])
            act_parts.append(self.actions[i])
            ret_parts.append(self.returns[i])
        if n_demo > 0:
            j = self._prioritized(self.d_returns[: self.d_size], n_demo, rng)
            obs_parts.append(self.d_obs[j])
            act_parts.append(self.d_actions[j])
            ret_parts.append(self.d_returns[j])
        return (
            np.concatenate(obs_parts),
            np.concatenate(act_parts),
            np.concatenate(ret_parts),
        )


def _seed_from_trajectories(
    buffer: SILBuffer, paths: list[str], gamma: float, reward_scale: float, transform: str
) -> int:
    """Replay deep Go-Explore trajectories into the buffer as expert returns."""
    import gymnasium as gym
    from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

    from dkrl.explore import _sticky_effective

    seeded = 0
    for path in paths:
        with Path(path).open("rb") as f:
            payload = pickle.load(f)
        sticky = float(payload.get("sticky", 0.0))
        score, actions = max(payload["trajectories"], key=lambda t: t[0])
        base = gym.make(
            "ALE/DonkeyKong-v5", frameskip=1, repeat_action_probability=0.0, full_action_space=True
        )
        env = AtariPreprocessing(
            base, noop_max=0, frame_skip=4, screen_size=84, grayscale_obs=True
        )
        env = FrameStackObservation(env, 4)
        obs, _ = env.reset(seed=payload["start_seed"])
        ep_obs, ep_act, ep_rew = [], [], []
        p_eff = 0
        for i, a in enumerate(actions):
            eff = _sticky_effective(int(a), p_eff, i, sticky)
            ep_obs.append(np.asarray(obs, dtype=np.uint8))
            ep_act.append(eff)
            obs, reward, terminated, truncated, _ = env.step(eff)
            p_eff = eff
            r = float(reward) * reward_scale if transform == "scale" else float(reward)
            ep_rew.append(r)
            if terminated or truncated:
                break
        env.close()
        returns = _discounted_returns(ep_rew, gamma)
        buffer.add_demos(ep_obs, ep_act, returns)
        seeded += len(ep_act)
        logger.info(f"seeded SIL buffer from {path}: score {score:.0f}, {len(ep_act)} transitions")
    return seeded


def _seed_from_self_demos(
    buffer: SILBuffer, paths: list[str], gamma: float, reward_scale: float, transform: str
) -> int:
    """Seed the demo buffer from the agent's own harvested episodes.

    Each path is a pickle written by ``harvest_self_trajectories`` holding stored
    observations, actions, and per-step rewards (no replay needed). Returns are
    computed as discounted return-to-go, matching the trajectory seeding.
    """
    seeded = 0
    for path in paths:
        with Path(path).open("rb") as f:
            payload = pickle.load(f)
        for ep in payload["episodes"]:
            ep_obs = list(np.asarray(ep["obs"], dtype=np.uint8))
            ep_act = [int(a) for a in ep["actions"]]
            rew = [
                (float(r) * reward_scale if transform == "scale" else float(r))
                for r in ep["rewards"]
            ]
            returns = _discounted_returns(rew, gamma)
            buffer.add_demos(ep_obs, ep_act, returns)
            seeded += len(ep_act)
        scores = [e["return"] for e in payload["episodes"]]
        logger.info(
            f"seeded SIL buffer from {path}: {len(payload['episodes'])} episodes, "
            f"returns {min(scores):.0f}..{max(scores):.0f}, {seeded} transitions total"
        )
    return seeded


def _discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    """Monte-Carlo discounted return-to-go for an episode."""
    out = [0.0] * len(rewards)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        out[t] = running
    return out


def train_sil(cfg: ExperimentConfig) -> dict:
    """Run PPO+SIL training to completion and return a ledger row.

    Args:
        cfg: Config with algo ``ppo_sil``.

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
    agent = PPOAgent(n_actions, cfg.env.frame_stack).to(device)
    sil = cfg.sil
    if cfg.env.reward_transform != "scale":
        raise ValueError(
            "ppo_sil requires env.reward_transform='scale': demo seeding computes returns from "
            "raw reward, which only matches the on-policy critic under the scale transform."
        )
    if sil.init_from:
        init_ckpt = torch.load(sil.init_from, map_location="cpu", weights_only=False)
        agent.load_state_dict(init_ckpt["model_state_dict"])
        logger.info(f"warm-started from {sil.init_from}")
    optimizer = optim.Adam(agent.parameters(), lr=cfg.ppo.learning_rate, eps=1e-5)

    num_envs = cfg.env.num_envs
    num_steps = cfg.ppo.num_steps
    batch_size = num_envs * num_steps
    minibatch_size = batch_size // cfg.ppo.num_minibatches
    num_updates = cfg.total_timesteps // batch_size
    obs_shape = (cfg.env.frame_stack, cfg.env.img_size, cfg.env.img_size)

    buffer = SILBuffer(sil.buffer_size, obs_shape, sil.demo_buffer_size)
    rng = np.random.default_rng(cfg.seed)
    if sil.seed_trajectories:
        _seed_from_trajectories(
            buffer,
            sil.seed_trajectories,
            cfg.ppo.gamma,
            cfg.env.reward_scale,
            cfg.env.reward_transform,
        )
    if sil.seed_self_demos:
        _seed_from_self_demos(
            buffer,
            sil.seed_self_demos,
            cfg.ppo.gamma,
            cfg.env.reward_scale,
            cfg.env.reward_transform,
        )

    obs = torch.zeros((num_steps, num_envs, *obs_shape), dtype=torch.uint8, device=device)
    actions = torch.zeros((num_steps, num_envs), dtype=torch.long, device=device)
    logprobs = torch.zeros((num_steps, num_envs), device=device)
    rewards = torch.zeros((num_steps, num_envs), device=device)
    dones = torch.zeros((num_steps, num_envs), device=device)
    values = torch.zeros((num_steps, num_envs), device=device)

    # Per-env episode accumulators for SIL returns (scaled reward, raw obs/action).
    ep_obs: list[list[np.ndarray]] = [[] for _ in range(num_envs)]
    ep_act: list[list[int]] = [[] for _ in range(num_envs)]
    ep_rew: list[list[float]] = [[] for _ in range(num_envs)]

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
            "sil_size",
        ]
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
            scaled = (
                reward * cfg.env.reward_scale if cfg.env.reward_transform == "scale" else reward
            )
            obs_np = next_obs.cpu().numpy()
            act_np = action.cpu().numpy()
            for i in range(num_envs):
                ep_obs[i].append(obs_np[i].astype(np.uint8))
                ep_act[i].append(int(act_np[i]))
                ep_rew[i].append(float(scaled[i]))
            rewards[step] = torch.from_numpy(scaled).to(device)
            done_np = np.logical_or(terminated, truncated)
            next_obs = torch.from_numpy(next_obs_np).to(device)
            next_done = torch.from_numpy(done_np.astype(np.float32)).to(device)
            for i in np.nonzero(done_np)[0]:
                ep_return_raw = sum(ep_rew[i]) / (
                    cfg.env.reward_scale if cfg.env.reward_transform == "scale" else 1.0
                )
                if ep_return_raw >= sil.min_return:
                    buffer.add_episode(
                        ep_obs[i], ep_act[i], _discounted_returns(ep_rew[i], cfg.ppo.gamma)
                    )
                ep_obs[i], ep_act[i], ep_rew[i] = [], [], []
            for ep_return, _ep_len, ep_max in stats.update(terminated, truncated, info):
                episodes_done += 1
                cleared = float(ep_max >= clear_threshold)
                clears_done += int(cleared)
                recent_returns.append(ep_return)
                recent_clears.append(cleared)
                writer.add_scalar("charts/episodic_return", ep_return, global_step)
                writer.add_scalar("charts/episode_cleared", cleared, global_step)

        # GAE.
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages, returns = compute_gae(
                rewards, values, dones, next_value, next_done, cfg.ppo.gamma, cfg.ppo.gae_lambda
            )

        b_obs = obs.reshape((-1, *obs_shape))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_inds = np.arange(batch_size)
        for _ in range(cfg.ppo.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                mb = b_inds[start : start + minibatch_size]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb], b_actions[mb]
                )
                logratio = newlogprob - b_logprobs[mb]
                ratio = logratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                mb_adv = b_advantages[mb]
                if cfg.ppo.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg_loss = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * torch.clamp(ratio, 1 - cfg.ppo.clip_coef, 1 + cfg.ppo.clip_coef),
                ).mean()
                v_loss = 0.5 * ((newvalue.view(-1) - b_returns[mb]) ** 2).mean()
                loss = pg_loss - cfg.ppo.ent_coef * entropy.mean() + v_loss * cfg.ppo.vf_coef
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), cfg.ppo.max_grad_norm)
                optimizer.step()
            if cfg.ppo.target_kl is not None and approx_kl > cfg.ppo.target_kl:
                break

        # Self-imitation updates: pull toward high-return buffered transitions.
        if buffer.size > sil.batch_size or buffer.d_size > sil.batch_size:
            for _ in range(sil.updates_per_iter):
                s_obs, s_act, s_ret = buffer.sample(sil.batch_size, sil.demo_fraction, rng)
                t_obs = torch.from_numpy(s_obs).to(device)
                t_act = torch.from_numpy(s_act).to(device)
                t_ret = torch.from_numpy(s_ret).to(device)
                _, logp, _, val = agent.get_action_and_value(t_obs, t_act)
                adv = torch.clamp(t_ret - val.view(-1), min=0.0)
                sil_pg = -(logp * adv.detach()).mean()
                sil_v = sil.vf_coef * 0.5 * (adv**2).mean()
                sil_loss = sil_pg + sil_v
                optimizer.zero_grad()
                sil_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), cfg.ppo.max_grad_norm)
                optimizer.step()

        elapsed = time.time() - start_time
        sps = int(global_step / elapsed)
        mean_return = float(np.mean(recent_returns)) if recent_returns else float("nan")
        clear_rate_100 = float(np.mean(recent_clears)) if recent_clears else float("nan")
        if recent_returns and mean_return > best_return_window:
            best_return_window = mean_return
        writer.add_scalar("charts/SPS", sps, global_step)
        writer.add_scalar("charts/clear_rate_100", clear_rate_100, global_step)
        writer.add_scalar("charts/sil_buffer", buffer.size, global_step)
        metrics.writerow(
            [
                global_step,
                round(elapsed, 1),
                mean_return,
                episodes_done,
                sps,
                clear_rate_100,
                buffer.size,
            ]
        )
        metrics_file.flush()
        if update % 10 == 0 or update == num_updates:
            logger.info(
                f"update {update}/{num_updates} step={global_step} "
                f"return100={mean_return:.1f} clear100={clear_rate_100:.2f} "
                f"sil={buffer.size} episodes={episodes_done} sps={sps}"
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
    logger.info(f"sil training done: best_return100={best_return_window:.1f}")
    return row
