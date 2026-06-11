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

    p_video = sub.add_parser("video", help="Record an MP4 of a checkpoint playing")
    p_video.add_argument("--checkpoint", type=Path, required=True)
    p_video.add_argument("--out", type=Path, required=True)
    p_video.add_argument("--episodes", type=int, default=1)
    p_video.add_argument("--seed", type=int, default=7)

    p_demos = sub.add_parser("collect-demos", help="Collect scripted screen-1 demonstrations")
    p_demos.add_argument("--out", type=Path, default=Path("data/demos_screen1.npz"))
    p_demos.add_argument("--episodes", type=int, default=25)
    p_demos.add_argument("--seed", type=int, default=7000)

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

    elif args.command == "collect-demos":
        from dkrl.demos import collect_demos
        from dkrl.ledger import git_sha
        from dkrl.log import setup_logging

        setup_logging("collect-demos")
        result = collect_demos(args.out, episodes=args.episodes, seed=args.seed)
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
