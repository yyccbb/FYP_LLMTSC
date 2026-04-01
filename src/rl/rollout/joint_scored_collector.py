from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from src.rl.prompting.parser import ACTION_TO_ID, parse_signal
from src.rl.rollout.collector import RolloutResult
from src.rl.training.distributed import DistributedRuntime


class JointScoredRolloutCollector:
    """Collect rollout records using joint candidate scoring over snapshot forks."""

    def __init__(
        self,
        env,
        tokenizer,
        model,
        k: int = 8,
        invalid_action_reward: float = 0.0,
        horizon: int = 3,
        discount_gamma: float = 0.99,
        reward_mode: str = "total_incoming_queue_length",
        reward_mix: Optional[Dict[str, float]] = None,
        continuation_policy: str = "resample_each_step",
        max_prompt_length: int = 2048,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        do_sample: bool = True,
        prompt_builder: Optional[Callable[[Any], List[str]]] = None,
        response_generator: Optional[Callable[[List[str], int], List[List[str]]]] = None,
        distributed: Optional[DistributedRuntime] = None,
    ) -> None:
        self.env = env
        self.tokenizer = tokenizer
        self.model = model
        self.k = int(k)
        self.invalid_action_reward = float(invalid_action_reward)
        self.horizon = int(horizon)
        self.discount_gamma = float(discount_gamma)
        self.reward_mode = self._normalize_reward_mode(reward_mode)
        self.continuation_policy = str(continuation_policy)
        self.max_prompt_length = int(max_prompt_length)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.do_sample = bool(do_sample)
        self.num_actions = len(ACTION_TO_ID)
        self.fallback_action = 0
        self._inter_name_to_index: Dict[str, int] = {}
        self.distributed = distributed or DistributedRuntime(
            enabled=False,
            world_size=1,
            process_index=0,
            local_process_index=0,
            device=None,
        )

        if self.k < 1:
            raise ValueError("rollout.k must be >= 1.")
        if self.horizon < 1:
            raise ValueError("rollout.horizon must be >= 1.")
        if self.continuation_policy != "resample_each_step":
            raise ValueError(
                f"Unsupported continuation_policy={self.continuation_policy}. "
                "Only 'resample_each_step' is supported."
            )
        if self.distributed.enabled and self.k % int(self.distributed.world_size) != 0:
            raise ValueError(
                f"rollout.k={self.k} must be divisible by world_size={self.distributed.world_size}."
            )

        mix = dict(reward_mix or {})
        self.alpha = float(mix.get("alpha", 0.6))
        self.beta = float(mix.get("beta", 0.3))
        if mix.get("global_weight") is None:
            self.global_weight = 1.0 - self.alpha - self.beta
        else:
            self.global_weight = float(mix["global_weight"])
        if self.global_weight < 0:
            raise ValueError("rollout.reward_mix.global_weight cannot be negative.")

        if prompt_builder is None:
            from src.rl.prompting.state_formatter import build_prompts_from_env

            self.prompt_builder = build_prompts_from_env
        else:
            self.prompt_builder = prompt_builder
        self.response_generator = response_generator

    @staticmethod
    def _normalize_reward_mode(value: str) -> str:
        normalized = str(value).strip().lower()
        aliases = {
            "total_incoming_queue_length": "total_incoming_queue_length",
            "incoming_queue": "total_incoming_queue_length",
            "queue_length": "total_incoming_queue_length",
            "total_pressure": "total_pressure",
            "pressure": "total_pressure",
        }
        if normalized not in aliases:
            raise ValueError(
                "rollout.reward_mode must be one of: "
                "total_incoming_queue_length, total_pressure."
            )
        return aliases[normalized]

    def _num_local_candidates(self) -> int:
        if not self.distributed.enabled:
            return self.k
        world_size = int(self.distributed.world_size)
        if self.k % world_size != 0:
            raise ValueError(f"rollout.k={self.k} must be divisible by world_size={world_size}.")
        return int(self.k // world_size)

    def _refresh_intersection_index(self) -> None:
        self._inter_name_to_index = {
            inter.inter_name: idx for idx, inter in enumerate(self.env.list_intersection)
        }

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

    def _generate_grouped_responses(
        self,
        prompts: List[str],
        num_return_sequences: int,
    ) -> List[List[str]]:
        if self.response_generator is not None:
            return self.response_generator(prompts, int(num_return_sequences))

        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "torch is required for model-based generation in JointScoredRolloutCollector."
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
                num_return_sequences=int(num_return_sequences),
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
            )

        decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        expected = len(prompts) * int(num_return_sequences)
        if len(decoded) != expected:
            raise RuntimeError(f"Expected {expected} decoded outputs, got {len(decoded)}.")

        grouped = [
            decoded[i * int(num_return_sequences) : (i + 1) * int(num_return_sequences)]
            for i in range(len(prompts))
        ]
        return self._strip_prompt_prefix(prompts, grouped)

    def _parse_actions(self, grouped_completions: List[List[str]]) -> List[List[Optional[int]]]:
        parsed_by_intersection: List[List[Optional[int]]] = []
        for completions in grouped_completions:
            parsed_by_intersection.append([parse_signal(completion) for completion in completions])
        return parsed_by_intersection

    def _build_joint_candidates(
        self,
        parsed_by_intersection: Sequence[Sequence[Optional[int]]],
    ) -> List[List[int]]:
        if len(parsed_by_intersection) == 0:
            return []

        num_candidates = len(parsed_by_intersection[0])
        for parsed in parsed_by_intersection:
            if len(parsed) != num_candidates:
                raise ValueError("Each intersection must have the same number of candidates.")

        num_intersections = len(parsed_by_intersection)
        candidates: List[List[int]] = []
        for candidate_id in range(num_candidates):
            candidate_actions: List[int] = []
            for intersection_id in range(num_intersections):
                parsed_action = parsed_by_intersection[intersection_id][candidate_id]
                is_valid = (
                    parsed_action is not None
                    and int(parsed_action) >= 0
                    and int(parsed_action) < self.num_actions
                )
                candidate_actions.append(int(parsed_action) if is_valid else self.fallback_action)
            candidates.append(candidate_actions)
        return candidates

    def _compute_local_metrics(self) -> List[float]:
        local_metrics: List[float] = []
        for inter in self.env.list_intersection:
            if self.reward_mode == "total_incoming_queue_length":
                values = inter.dic_feature.get("lane_num_waiting_vehicle_in", [])
                local_metrics.append(float(np.sum(np.asarray(values, dtype=float))))
            else:
                values = inter.dic_feature.get("pressure", [])
                local_metrics.append(float(abs(np.sum(np.asarray(values, dtype=float)))))
        return local_metrics

    def _compute_network_reward(self, local_metrics: Sequence[float]) -> float:
        if len(local_metrics) == 0:
            return 0.0

        global_mean = float(np.mean(np.asarray(local_metrics, dtype=float)))
        mixed_costs: List[float] = []
        for idx, inter in enumerate(self.env.list_intersection):
            local_value = float(local_metrics[idx])
            neighbor_ids = getattr(inter, "neighbor_ENWS", None) or []
            neighbor_values = [
                float(local_metrics[self._inter_name_to_index[neighbor_id]])
                for neighbor_id in neighbor_ids
                if neighbor_id is not None and neighbor_id in self._inter_name_to_index
            ]
            neighbor_mean = float(np.mean(np.asarray(neighbor_values, dtype=float))) if neighbor_values else local_value
            mixed_cost = (
                self.alpha * local_value
                + self.beta * neighbor_mean
                + self.global_weight * global_mean
            )
            mixed_costs.append(float(mixed_cost))

        return -float(np.mean(np.asarray(mixed_costs, dtype=float)))

    def _resample_joint_action(self) -> List[int]:
        prompts = self.prompt_builder(self.env)
        grouped = self._generate_grouped_responses(prompts, num_return_sequences=1)
        actions: List[int] = []
        for completions in grouped:
            parsed_action = parse_signal(completions[0]) if len(completions) > 0 else None
            if parsed_action is None or parsed_action < 0 or parsed_action >= self.num_actions:
                actions.append(self.fallback_action)
            else:
                actions.append(int(parsed_action))
        return actions

    def _evaluate_joint_candidate(self, root_snapshot, initial_action: Sequence[int]) -> float:
        self.env.load_snapshot(root_snapshot)
        joint_action = [int(action_id) for action_id in initial_action]
        discounted_return = 0.0
        discount = 1.0
        done = False

        for step in range(self.horizon):
            if step > 0:
                joint_action = self._resample_joint_action()

            _next_state, _reward, done, _avg_reward = self.env.step(joint_action)
            step_reward = self._compute_network_reward(self._compute_local_metrics())
            discounted_return += discount * float(step_reward)
            discount *= self.discount_gamma
            if done:
                break

        return float(discounted_return)

    def _project_candidate_scores(
        self,
        parsed_by_intersection: Sequence[Sequence[Optional[int]]],
        candidate_returns: Sequence[float],
    ) -> List[List[float]]:
        action_scores_by_intersection: List[List[float]] = []
        for parsed_actions in parsed_by_intersection:
            per_action_scores: List[float] = []
            for action_id in range(self.num_actions):
                returns = [
                    float(candidate_returns[candidate_idx])
                    for candidate_idx, parsed_action in enumerate(parsed_actions)
                    if parsed_action == action_id
                ]
                if len(returns) == 0:
                    per_action_scores.append(float(self.invalid_action_reward))
                else:
                    per_action_scores.append(float(np.mean(np.asarray(returns, dtype=float))))
            action_scores_by_intersection.append(per_action_scores)
        return action_scores_by_intersection

    def _gather_global_candidate_payload(
        self,
        local_candidate_returns: Sequence[float],
        local_parsed_by_intersection: Sequence[Sequence[Optional[int]]],
        local_joint_candidates: Sequence[Sequence[int]],
    ) -> tuple[List[float], List[List[Optional[int]]], List[List[int]]]:
        local_payload = {
            "candidate_returns": [float(v) for v in local_candidate_returns],
            "parsed_by_intersection": [list(v) for v in local_parsed_by_intersection],
            "joint_candidates": [list(v) for v in local_joint_candidates],
        }
        gathered_payloads = self.distributed.all_gather_object(local_payload)

        num_intersections = len(local_parsed_by_intersection)
        global_candidate_returns: List[float] = []
        global_parsed_by_intersection: List[List[Optional[int]]] = [[] for _ in range(num_intersections)]
        global_joint_candidates: List[List[int]] = []

        for payload in gathered_payloads:
            rank_returns = list(payload.get("candidate_returns", []))
            rank_parsed = list(payload.get("parsed_by_intersection", []))
            rank_joint_candidates = list(payload.get("joint_candidates", []))

            if len(rank_returns) != len(rank_joint_candidates):
                raise RuntimeError(
                    "Mismatched gathered candidate payload lengths for returns and joint actions."
                )
            if len(rank_parsed) != num_intersections:
                raise RuntimeError(
                    "Mismatched gathered candidate payload lengths for parsed intersection actions."
                )

            for intersection_id in range(num_intersections):
                global_parsed_by_intersection[intersection_id].extend(rank_parsed[intersection_id])
            global_candidate_returns.extend(rank_returns)
            global_joint_candidates.extend(rank_joint_candidates)

        return global_candidate_returns, global_parsed_by_intersection, global_joint_candidates

    def collect_episode(self, run_counts: int, action_interval: int) -> RolloutResult:
        if run_counts % action_interval != 0:
            raise ValueError(
                f"RUN_COUNTS={run_counts} must be divisible by action_interval={action_interval}."
            )

        done = False
        _ = self.env.reset()
        self._refresh_intersection_index()
        local_k = self._num_local_candidates()

        total_steps = int(run_counts / action_interval)
        dataset_records: List[Dict[str, Any]] = []
        selected_actions_by_step: List[List[int]] = []
        selected_rewards_by_step: List[List[float]] = []

        for step_id in range(total_steps):
            if done:
                break

            prompts = self.prompt_builder(self.env)
            local_grouped_completions = self._generate_grouped_responses(
                prompts,
                num_return_sequences=local_k,
            )
            local_parsed_by_intersection = self._parse_actions(local_grouped_completions)
            local_joint_candidates = self._build_joint_candidates(local_parsed_by_intersection)

            root_snapshot = self.env.capture_snapshot()
            local_candidate_returns = [
                self._evaluate_joint_candidate(root_snapshot, joint_action)
                for joint_action in local_joint_candidates
            ]

            (
                global_candidate_returns,
                global_parsed_by_intersection,
                global_joint_candidates,
            ) = self._gather_global_candidate_payload(
                local_candidate_returns=local_candidate_returns,
                local_parsed_by_intersection=local_parsed_by_intersection,
                local_joint_candidates=local_joint_candidates,
            )

            if len(global_candidate_returns) != self.k:
                raise RuntimeError(
                    f"Expected {self.k} total candidates, got {len(global_candidate_returns)}."
                )
            if len(global_joint_candidates) != self.k:
                raise RuntimeError(
                    f"Expected {self.k} total joint action candidates, got {len(global_joint_candidates)}."
                )

            action_scores_by_intersection = self._project_candidate_scores(
                global_parsed_by_intersection,
                global_candidate_returns,
            )

            selected_idx: Optional[int] = None
            if self.distributed.is_main_process:
                selected_idx = int(np.random.randint(0, len(global_candidate_returns)))
            selected_idx = int(self.distributed.broadcast_object(selected_idx))
            selected_joint_action = [int(v) for v in global_joint_candidates[selected_idx]]

            self.env.load_snapshot(root_snapshot)
            _next_state, _reward, done, _avg_reward = self.env.step(selected_joint_action)

            selected_actions_by_step.append([int(v) for v in selected_joint_action])
            selected_rewards_by_step.append(
                [
                    float(action_scores_by_intersection[intersection_id][selected_joint_action[intersection_id]])
                    for intersection_id in range(len(selected_joint_action))
                ]
            )

            for intersection_id, prompt in enumerate(prompts):
                dataset_records.append(
                    {
                        "prompt": prompt,
                        "q_values": [float(v) for v in action_scores_by_intersection[intersection_id]],
                        "step_id": int(step_id),
                        "intersection_id": int(intersection_id),
                    }
                )

        return RolloutResult(
            dataset_records=dataset_records,
            selected_actions_by_step=selected_actions_by_step,
            selected_rewards_by_step=selected_rewards_by_step,
        )
