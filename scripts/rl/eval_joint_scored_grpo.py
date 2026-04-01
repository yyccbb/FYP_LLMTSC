import argparse
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.rl.pipeline.joint_scored_grpo_pipeline import JointScoredGRPOPipeline, load_experiment_config


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate joint-scored GRPO rollout behavior.")
    parser.add_argument("--config", type=str, default="src/rl/configs/grpo/joint_scored_base.yaml")
    parser.add_argument(
        "--override",
        type=str,
        action="append",
        default=[],
        help="Optional override YAML files. Can be specified multiple times.",
    )
    parser.add_argument("--episodes", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_experiment_config(args.config, args.override)

    pipeline = JointScoredGRPOPipeline(config)
    rollout_results = pipeline.evaluate(episodes=args.episodes)

    summary = [
        {
            "episode": idx,
            "steps": result.num_steps,
            "dataset_records": len(result.dataset_records),
        }
        for idx, result in enumerate(rollout_results)
    ]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
