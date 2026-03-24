import argparse
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.utils.utils import run_debugpy_server
from src.rl.pipeline.qguided_grpo_pipeline import QGuidedGRPOPipeline, deep_update, load_experiment_config


def parse_args():
    parser = argparse.ArgumentParser(description="Train Q-guided GRPO policy.")
    parser.add_argument("--config", type=str, default="src/rl/configs/grpo/base.yaml")
    parser.add_argument(
        "--override",
        type=str,
        action="append",
        default=[],
        help="Optional override YAML files. Can be specified multiple times.",
    )
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--llm_path", type=str, default=None)
    parser.add_argument("--debug", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.debug:
        run_debugpy_server()
    config = load_experiment_config(args.config, args.override)

    cli_override = {}
    if args.episodes is not None:
        cli_override = deep_update(cli_override, {"train": {"episodes": args.episodes}})
    if args.llm_path is not None:
        cli_override = deep_update(cli_override, {"policy": {"llm_path": args.llm_path}})
    if cli_override:
        config = deep_update(config, cli_override)

    pipeline = QGuidedGRPOPipeline(config)
    logs = pipeline.run()
    print(json.dumps(logs, indent=2))


if __name__ == "__main__":
    main()
