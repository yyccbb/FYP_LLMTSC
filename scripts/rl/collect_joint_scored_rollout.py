import argparse
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.rl.pipeline.joint_scored_grpo_pipeline import (
    JointScoredGRPOPipeline,
    deep_update,
    load_experiment_config,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Collect one rollout artifact for joint-scored GRPO.")
    parser.add_argument("--config", type=str, default="src/rl/configs/grpo/joint_scored_base.yaml")
    parser.add_argument(
        "--override",
        type=str,
        action="append",
        default=[],
        help="Optional override YAML files. Can be specified multiple times.",
    )
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--llm_path", type=str, default=None)
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--proj_name", type=str, default=None)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--debug", action="store_true", default=False)
    return parser.parse_args()


def build_cli_override(args) -> dict:
    cli_override = {}
    if args.episodes is not None:
        cli_override = deep_update(cli_override, {"train": {"episodes": args.episodes}})
    if args.llm_path is not None:
        cli_override = deep_update(cli_override, {"policy": {"llm_path": args.llm_path}})
    if args.proj_name is not None:
        cli_override = deep_update(cli_override, {"env": {"project_name": args.proj_name}})
    if args.wandb:
        cli_override = deep_update(
            cli_override,
            {
                "logging": {"wandb": {"enabled": True}},
                "grpo": {"report_to": "wandb"},
            },
        )
    cli_override = deep_update(cli_override, {"paths": {"run_name": args.run_name}})
    return cli_override


def main():
    args = parse_args()
    if args.debug:
        from src.utils.utils import run_debugpy_server

        run_debugpy_server(port=5678)
    config = load_experiment_config(args.config, args.override)

    cli_override = build_cli_override(args)
    if cli_override:
        config = deep_update(config, cli_override)

    pipeline = JointScoredGRPOPipeline(config)
    summary = pipeline.collect_rollout_once()
    if pipeline.is_main_process:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
