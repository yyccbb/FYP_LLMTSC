from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.rl.prompting.parser import parse_signal


@dataclass
class RolloutResult:
    dataset_records: List[Dict[str, Any]]
    selected_actions_by_step: List[List[int]]
    selected_rewards_by_step: List[List[float]]

    @property
    def num_steps(self) -> int:
        return len(self.selected_actions_by_step)


class QGuidedRolloutCollector:
    """Collect one rollout episode using K sampled completions and top-Q action execution."""

    def __init__(
        self,
        env,
        tokenizer,
        model,
        q_rewarder,
        k: int = 4,
        invalid_action_reward: float = 0.0,
        max_prompt_length: int = 2048,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        do_sample: bool = True,
        prompt_builder: Optional[Callable[[Any], List[str]]] = None,
        response_generator: Optional[Callable[[List[str], int], List[List[str]]]] = None,
    ) -> None:
        self.env = env
        self.tokenizer = tokenizer
        self.model = model
        self.q_rewarder = q_rewarder
        self.k = k
        self.invalid_action_reward = invalid_action_reward
        self.max_prompt_length = max_prompt_length
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.do_sample = do_sample
        if prompt_builder is None:
            from src.rl.prompting.state_formatter import build_prompts_from_env

            self.prompt_builder = build_prompts_from_env
        else:
            self.prompt_builder = prompt_builder
        self.response_generator = response_generator

    @staticmethod
    def map_actions_to_rewards(
        action_ids: Sequence[Optional[int]],
        q_values: Sequence[float],
        invalid_action_reward: float,
    ) -> List[float]:
        rewards: List[float] = []
        for action_id in action_ids:
            if action_id is None or action_id < 0 or action_id >= len(q_values):
                rewards.append(float(invalid_action_reward))
            else:
                rewards.append(float(q_values[action_id]))
        return rewards

    @staticmethod
    def select_top_action(
        action_ids: Sequence[Optional[int]],
        rewards: Sequence[float],
        fallback_action: int = 0,
    ) -> Tuple[int, int]:
        if len(rewards) == 0:
            return fallback_action, -1

        best_idx = int(np.argmax(np.asarray(rewards)))
        best_action = action_ids[best_idx]
        if best_action is None:
            return fallback_action, best_idx
        return int(best_action), best_idx

    def _strip_prompt_prefix(self, prompts: Sequence[str], grouped_decoded: List[List[str]]) -> List[List[str]]:
        completions: List[List[str]] = []
        for prompt, decoded_group in zip(prompts, grouped_decoded):
            group: List[str] = []
            for decoded in decoded_group:
                if decoded.startswith(prompt):
                    group.append(decoded[len(prompt) :])
                else:
                    group.append(decoded)
            completions.append(group)
        return completions

    def _generate_grouped_responses(self, prompts: List[str]) -> List[List[str]]:
        if self.response_generator is not None:
            return self.response_generator(prompts, self.k)

        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "torch is required for model-based generation in QGuidedRolloutCollector."
            ) from exc

        inputs = self.tokenizer(
            prompts,
            truncation=True,
            max_length=self.max_prompt_length,
            padding=True,
            return_tensors="pt",
        )

        model_device = next(self.model.parameters()).device
        for key, value in list(inputs.items()):
            if isinstance(value, torch.Tensor):
                inputs[key] = value.to(model_device)

        with torch.no_grad():
            generated = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                do_sample=self.do_sample,
                top_k=self.top_k,
                top_p=self.top_p,
                temperature=self.temperature,
                num_return_sequences=self.k,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
            )

        decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        if len(decoded) != len(prompts) * self.k:
            raise RuntimeError(
                f"Expected {len(prompts) * self.k} decoded outputs, got {len(decoded)}."
            )

        grouped = [decoded[i * self.k : (i + 1) * self.k] for i in range(len(prompts))]
        return self._strip_prompt_prefix(prompts, grouped)

    def collect_episode(self, run_counts: int, action_interval: int) -> RolloutResult:
        if run_counts % action_interval != 0:
            raise ValueError(
                f"RUN_COUNTS={run_counts} must be divisible by action_interval={action_interval}."
            )

        done = False
        _ = self.env.reset()

        total_steps = int(run_counts / action_interval)
        dataset_records: List[Dict[str, Any]] = []
        selected_actions_by_step: List[List[int]] = []
        selected_rewards_by_step: List[List[float]] = []

        for step_id in range(total_steps): # TODO: warmup
            if done:
                break

            prompts = self.prompt_builder(self.env)
            grouped_completions = self._generate_grouped_responses(prompts)

            critic_state, _ = self.env.get_state(self.q_rewarder.list_state_features)
            q_values = self.q_rewarder.get_q_values(critic_state)

            action_list: List[int] = []
            reward_list: List[float] = []
            for intersection_id, prompt in enumerate(prompts):
                completions = grouped_completions[intersection_id]
                sampled_action_ids = [parse_signal(completion) for completion in completions]
                sampled_rewards = self.map_actions_to_rewards(
                    sampled_action_ids,
                    q_values[intersection_id],
                    self.invalid_action_reward,
                )
                selected_action, _selected_idx = self.select_top_action( # TODO: random?
                    sampled_action_ids,
                    sampled_rewards,
                    fallback_action=0,
                )

                action_list.append(selected_action)
                reward_list.append(float(q_values[intersection_id][selected_action]))

                dataset_records.append( # TODO: no completion??
                    {
                        "prompt": prompt,
                        "q_values": [float(v) for v in q_values[intersection_id]],
                        "step_id": int(step_id),
                        "intersection_id": int(intersection_id),
                    }
                )

            _next_state, _reward, done, _ = self.env.step(action_list)
            selected_actions_by_step.append(action_list)
            selected_rewards_by_step.append(reward_list)

        return RolloutResult(
            dataset_records=dataset_records,
            selected_actions_by_step=selected_actions_by_step,
            selected_rewards_by_step=selected_rewards_by_step,
        )
