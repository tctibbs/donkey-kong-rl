"""Command-line entry point: train, evaluate, random-baseline, plot."""

import argparse
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

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

    args = parser.parse_args()

    if args.command == "train":
        from dkrl.train.ppo import train

        cfg = ExperimentConfig.from_yaml(args.config)
        if args.seed is not None:
            cfg = cfg.model_copy(update={"seed": args.seed})
        if args.total_timesteps is not None:
            cfg = cfg.model_copy(update={"total_timesteps": args.total_timesteps})
        if args.no_track:
            cfg = cfg.model_copy(update={"track": False})
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
        from dkrl.plots import plot_learning_curves

        out = plot_learning_curves(args.metrics, args.out)
        print(out)
