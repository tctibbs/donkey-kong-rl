"""Command-line entry point: train, evaluate, random-baseline, plot, video."""

import argparse
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from dkrl.config import EvalConfig, ExperimentConfig
from dkrl.ledger import append_row


def main() -> None:
    """Parse CLI arguments and dispatch to the requested command."""
    load_dotenv()
    parser = argparse.ArgumentParser(prog="dkrl", description="Donkey Kong RL experiments")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train an agent from a YAML config")
    p_train.add_argument("--config", type=Path, required=True)
    p_train.add_argument("--seed", type=int, default=None, help="Override config seed")
    p_train.add_argument(
        "--total-timesteps", type=int, default=None, help="Override total timesteps"
    )
    p_train.add_argument("--no-track", action="store_true", help="Disable W&B tracking")

    p_eval = sub.add_parser("evaluate", help="Evaluate a checkpoint (Machado protocol)")
    p_eval.add_argument("--checkpoint", type=Path, required=True)
    p_eval.add_argument("--episodes", type=int, default=None)
    p_eval.add_argument("--seed", type=int, default=None)

    p_rand = sub.add_parser("random-baseline", help="Measure the random-policy baseline")
    p_rand.add_argument("--env-id", type=str, default="DonkeyKong-v5")
    p_rand.add_argument("--episodes", type=int, default=100)
    p_rand.add_argument("--seed", type=int, default=1000)

    p_plot = sub.add_parser("plot", help="Plot learning curves from metrics CSVs")
    p_plot.add_argument("metrics", type=Path, nargs="+")
    p_plot.add_argument("--out", type=Path, default=Path("results/plots/learning_curves.png"))

    p_stats = sub.add_parser("stats", help="Aggregate eval rows (IQM, bootstrap CIs)")
    p_stats.add_argument("--name", type=str, required=True, help="Run-id prefix to aggregate")

    p_video = sub.add_parser("video", help="Record an MP4 of a checkpoint playing")
    p_video.add_argument("--checkpoint", type=Path, required=True)
    p_video.add_argument("--out", type=Path, required=True)
    p_video.add_argument("--episodes", type=int, default=1)
    p_video.add_argument("--seed", type=int, default=7)

    p_sdemos = sub.add_parser(
        "self-demos", help="Harvest a trained agent's best episodes as demos"
    )
    p_sdemos.add_argument("--checkpoint", type=Path, required=True)
    p_sdemos.add_argument("--out", type=Path, required=True)
    p_sdemos.add_argument("--episodes", type=int, default=300)
    p_sdemos.add_argument("--keep", type=int, default=25)
    p_sdemos.add_argument("--seed", type=int, default=11000)
    p_sdemos.add_argument("--merge-with", type=Path, default=None)

    p_exp = sub.add_parser("explore", help="Go-Explore trajectory search (demo generation)")
    p_exp.add_argument("--out", type=Path, required=True)
    p_exp.add_argument("--iterations", type=int, default=100000)
    p_exp.add_argument("--seed", type=int, default=0)
    p_exp.add_argument("--target-score", type=float, default=35000.0)
    p_exp.add_argument("--resume-from", type=Path, nargs="*", default=None)

    p_rep = sub.add_parser("replay-demos", help="Convert explored trajectories to BC demos")
    p_rep.add_argument("--trajectories", type=Path, nargs="+", required=True)
    p_rep.add_argument("--out", type=Path, required=True)
    p_rep.add_argument("--top-n", type=int, default=25)
    p_rep.add_argument("--min-score", type=float, default=6000.0)
    p_rep.add_argument("--merge-with", type=Path, default=None)

    p_demos = sub.add_parser("collect-demos", help="Collect scripted screen-1 demonstrations")
    p_demos.add_argument("--out", type=Path, default=Path("data/demos_screen1.npz"))
    p_demos.add_argument("--episodes", type=int, default=25)
    p_demos.add_argument("--seed", type=int, default=7000)
    p_demos.add_argument("--sticky", type=float, default=0.0)

    args = parser.parse_args()

    if args.command == "train":
        cfg = ExperimentConfig.from_yaml(args.config)
        if args.seed is not None:
            cfg = cfg.model_copy(update={"seed": args.seed})
        if args.total_timesteps is not None:
            cfg = cfg.model_copy(update={"total_timesteps": args.total_timesteps})
        if args.no_track:
            cfg = cfg.model_copy(update={"track": False})
        if cfg.algo == "ppo_rnd":
            from dkrl.train.ppo_rnd import train_rnd as train
        else:
            from dkrl.train.ppo import train
        row = train(cfg)
        append_row(row)

    elif args.command == "evaluate":
        from dkrl.evaluate import evaluate_checkpoint
        from dkrl.ledger import git_sha
        from dkrl.log import setup_logging

        setup_logging("eval")
        result = evaluate_checkpoint(args.checkpoint, episodes=args.episodes, seed=args.seed)
        result["git_sha"] = git_sha()
        result["finished_at"] = datetime.now(UTC).isoformat(timespec="seconds")
        append_row(result)

    elif args.command == "random-baseline":
        from dkrl.evaluate import random_baseline
        from dkrl.ledger import git_sha
        from dkrl.log import setup_logging

        setup_logging("random-baseline")
        result = random_baseline(
            EvalConfig(), env_id=args.env_id, episodes=args.episodes, seed=args.seed
        )
        result["git_sha"] = git_sha()
        result["finished_at"] = datetime.now(UTC).isoformat(timespec="seconds")
        append_row(result)

    elif args.command == "plot":
        from dkrl.log import setup_logging
        from dkrl.plots import plot_learning_curves

        setup_logging("plot")
        out = plot_learning_curves(args.metrics, args.out)
        logger.info(f"plot written to {out}")

    elif args.command == "stats":
        import json

        from dkrl.log import setup_logging
        from dkrl.stats import aggregate_runs

        setup_logging("stats")
        result = aggregate_runs(args.name)
        logger.info(json.dumps(result, indent=2))

    elif args.command == "self-demos":
        from dkrl.demos import collect_self_demos
        from dkrl.ledger import git_sha
        from dkrl.log import setup_logging

        setup_logging("self-demos")
        result = collect_self_demos(
            args.checkpoint,
            args.out,
            episodes=args.episodes,
            keep=args.keep,
            seed=args.seed,
            merge_with=args.merge_with,
        )
        result.update(
            {
                "kind": "self_demos",
                "checkpoint": str(args.checkpoint),
                "seed": args.seed,
                "git_sha": git_sha(),
                "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
            }
        )
        append_row(result)

    elif args.command == "explore":
        from dkrl.explore import explore
        from dkrl.ledger import git_sha
        from dkrl.log import setup_logging

        setup_logging(f"explore-s{args.seed}")
        result = explore(
            args.out,
            iterations=args.iterations,
            seed=args.seed,
            target_score=args.target_score,
            resume_from=args.resume_from,
        )
        result.update(
            {
                "kind": "explore",
                "seed": args.seed,
                "git_sha": git_sha(),
                "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
            }
        )
        append_row(result)

    elif args.command == "replay-demos":
        from dkrl.explore import replay_to_demos
        from dkrl.ledger import git_sha
        from dkrl.log import setup_logging

        setup_logging("replay-demos")
        result = replay_to_demos(
            args.trajectories,
            args.out,
            top_n=args.top_n,
            min_score=args.min_score,
            merge_with=args.merge_with,
        )
        result.update(
            {
                "kind": "replay_demos",
                "git_sha": git_sha(),
                "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
            }
        )
        append_row(result)

    elif args.command == "collect-demos":
        from dkrl.demos import collect_demos
        from dkrl.ledger import git_sha
        from dkrl.log import setup_logging

        setup_logging("collect-demos")
        result = collect_demos(
            args.out, episodes=args.episodes, seed=args.seed, sticky_action_prob=args.sticky
        )
        result["sticky_action_prob"] = args.sticky
        result.update(
            {
                "kind": "demos",
                "seed": args.seed,
                "git_sha": git_sha(),
                "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
            }
        )
        append_row(result)

    elif args.command == "video":
        from dkrl.log import setup_logging
        from dkrl.video import record_video

        setup_logging("video")
        from dkrl.ledger import git_sha

        result = record_video(args.checkpoint, args.out, episodes=args.episodes, seed=args.seed)
        result.update(
            {
                "kind": "video",
                "checkpoint": str(args.checkpoint),
                "seed": args.seed,
                "git_sha": git_sha(),
                "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
            }
        )
        append_row(result)
