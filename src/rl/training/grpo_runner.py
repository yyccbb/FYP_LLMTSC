from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from src.rl.prompting.parser import parse_signal


def _completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        if "content" in completion:
            return str(completion["content"])
        return str(completion)
    if isinstance(completion, list):
        parts: List[str] = []
        for item in completion:
            if isinstance(item, dict) and "content" in item:
                parts.append(str(item["content"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(completion)


def build_qvalue_reward_func(invalid_action_reward: float = 0.0):
    """Create a TRL-compatible reward function using per-sample q_values."""

    def reward_func(completions, q_values, **kwargs):
        _ = kwargs
        rewards: List[float] = []
        for completion, q_value in zip(completions, q_values):
            text = _completion_to_text(completion)
            action_id = parse_signal(text)
            if action_id is None or action_id < 0 or action_id >= len(q_value):
                rewards.append(float(invalid_action_reward))
            else:
                rewards.append(float(q_value[action_id]))
        return rewards

    return reward_func


class GRPOTrainingRunner:
    """Episode-level GRPO optimization runner on collected prompt/qvalue records."""

    def __init__(self, grpo_config: Dict[str, Any], invalid_action_reward: float = 0.0) -> None:
        self.grpo_config = dict(grpo_config)
        self.invalid_action_reward = float(invalid_action_reward)

    def _build_grpo_args(self):
        try:
            from trl import GRPOConfig
        except ImportError as exc:
            raise ImportError(
                "trl is required for GRPO training. Install dependencies from requirements-rl.txt."
            ) from exc

        cfg = self.grpo_config
        return GRPOConfig(
            output_dir=cfg["output_dir"],
            learning_rate=float(cfg.get("learning_rate", 1e-6)),
            per_device_train_batch_size=int(cfg.get("per_device_train_batch_size", 1)),
            gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 1)),
            num_train_epochs=float(cfg.get("num_train_epochs", 1.0)),
            logging_steps=int(cfg.get("logging_steps", 1)),
            save_steps=int(cfg.get("save_steps", 1000)),
            save_total_limit=int(cfg.get("save_total_limit", 2)),
            report_to=cfg.get("report_to", "none"),
            num_generations=int(cfg["num_generations"]),
            max_completion_length=int(cfg.get("max_completion_length", 256)),
            temperature=float(cfg.get("temperature", 1.0)),
            top_p=float(cfg.get("top_p", 1.0)),
            top_k=int(cfg.get("top_k", 0)),
            scale_rewards=cfg.get("scale_rewards", "group"),
            beta=float(cfg.get("beta", 0.02)),
            num_iterations=int(cfg.get("num_iterations", 1)),
            remove_unused_columns=False,
            eval_strategy="no",
            save_strategy=cfg.get("save_strategy", "steps"),
        )

    def train_on_records(self, model, tokenizer, records: Iterable[Dict[str, Any]]):
        records = list(records)
        if len(records) == 0:
            return None

        try:
            from datasets import Dataset
            from trl import GRPOTrainer
        except ImportError as exc:
            raise ImportError(
                "datasets and trl are required for GRPO training. Install dependencies from requirements-rl.txt."
            ) from exc

        train_dataset = Dataset.from_list(records)
        reward_func = build_qvalue_reward_func(self.invalid_action_reward)
        grpo_args = self._build_grpo_args()

        trainer_kwargs = {
            "model": model,
            "args": grpo_args,
            "reward_funcs": reward_func,
            "train_dataset": train_dataset,
        }

        # TRL versions differ on whether they expect `processing_class` or `tokenizer`.
        try:
            trainer = GRPOTrainer(processing_class=tokenizer, **trainer_kwargs)
        except TypeError:
            trainer = GRPOTrainer(tokenizer=tokenizer, **trainer_kwargs)

        trainer.train()
        return trainer
