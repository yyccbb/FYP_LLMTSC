from __future__ import annotations

import copy
import os
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional

import yaml

from src.rl.rollout.collector import QGuidedRolloutCollector, RolloutResult
from src.rl.training.grpo_runner import GRPOTrainingRunner
from src.utils.config import DIC_BASE_AGENT_CONF, DIC_CITY_ALIASES, DIC_CITY_SPECS, DIC_TRAFFIC_ENV_CONF

if TYPE_CHECKING:
    from src.rl.critic.colight_q_rewarder import CoLightQRewarder


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def load_experiment_config(base_config: str | Path, override_configs: Optional[Iterable[str | Path]] = None) -> Dict[str, Any]:
    config = load_yaml(base_config)
    for override in override_configs or []:
        config = deep_update(config, load_yaml(override))
    return config


class QGuidedGRPOPipeline:
    """Q-guided GRPO training pipeline with episode-level optimization."""

    def __init__(
        self,
        config: Dict[str, Any],
        collector: Optional[QGuidedRolloutCollector] = None,
        grpo_runner: Optional[GRPOTrainingRunner] = None,
        env=None,
        tokenizer=None,
        policy_model=None,
        q_rewarder=None,
    ) -> None:
        self.config = copy.deepcopy(config)
        self._finalize_config_defaults()
        self._build_city_and_env_conf()

        self.dic_path = self._prepare_paths()
        self._copy_cityflow_files()

        self.env = env
        self.tokenizer = tokenizer
        self.policy_model = policy_model
        self.q_rewarder = q_rewarder

        if collector is None:
            if self.env is None:
                self.env = self._init_env()
            if self.policy_model is None or self.tokenizer is None:
                self.policy_model, self.tokenizer = self._init_policy_model()
            if self.q_rewarder is None:
                self.q_rewarder = self._init_q_rewarder()
            self.collector = self._init_collector()
        else:
            self.collector = collector

        if grpo_runner is None:
            self.grpo_runner = self._init_grpo_runner()
        else:
            self.grpo_runner = grpo_runner

    def _finalize_config_defaults(self) -> None:
        self.config.setdefault("paths", {})
        self.config.setdefault("env", {})
        self.config.setdefault("policy", {})
        self.config.setdefault("critic", {})
        self.config.setdefault("rollout", {})
        self.config.setdefault("grpo", {})
        self.config.setdefault("train", {})

        self.config["rollout"].setdefault("k", 4)
        self.config["rollout"].setdefault("invalid_action_reward", 0.0)

        self.config["grpo"].setdefault("scale_rewards", "group")
        self.config["grpo"].setdefault("beta", 0.02)
        self.config["grpo"].setdefault("num_iterations", 1)

        self.config["train"].setdefault("episodes", 1)
        self.config["train"].setdefault("save_every_episode", 1)

    def _build_city_and_env_conf(self) -> None:
        env_cfg = self.config["env"]
        raw_city = env_cfg.get("city")
        if not raw_city:
            raise ValueError(f"City is not provided.")
        city_name = DIC_CITY_ALIASES.get(str(raw_city).lower(), raw_city)
        if city_name not in DIC_CITY_SPECS:
            raise ValueError(f"Unsupported city '{raw_city}'.")

        city_specs = DIC_CITY_SPECS[city_name]
        self.city_name = city_name
        self.road_net = city_specs.road_net

        traffic_file = env_cfg.get("traffic_file")
        if not traffic_file:
            print(f"Traffic file is not provided. Using first in list_traffic_files...")
            traffic_file = city_specs.list_traffic_files[0]

        if traffic_file not in city_specs.list_traffic_files:
            raise ValueError(
                f"Unsupported traffic_file '{traffic_file}' for city '{city_name}'. "
                f"Available files: {', '.join(city_specs.list_traffic_files)}"
            )

        run_counts = int(env_cfg.get("run_counts", city_specs.count))
        action_interval = int(env_cfg.get("action_interval", DIC_TRAFFIC_ENV_CONF["MIN_ACTION_TIME"]))
        if run_counts % action_interval != 0:
            raise ValueError(
                f"RUN_COUNTS={run_counts} must be divisible by action_interval={action_interval}."
            )

        num_row = int(self.road_net.split("_")[0])
        num_col = int(self.road_net.split("_")[1])
        num_intersections = num_row * num_col

        env_extra = {
            "NUM_AGENTS": num_intersections, # TODO: 1?
            "NUM_INTERSECTIONS": num_intersections,
            "MODEL_NAME": env_cfg.get("model_name", "QGuidedGRPO"),
            "MODEL": env_cfg.get("model_name", "QGuidedGRPO"),
            "PROJECT_NAME": env_cfg.get("project_name", "QGuidedGRPO"),
            "RUN_COUNTS": run_counts,
            "NUM_ROW": num_row,
            "NUM_COL": num_col,
            "MIN_ACTION_TIME": action_interval,
            "MEASURE_TIME": action_interval,
            "TRAFFIC_FILE": traffic_file,
            "ROADNET_FILE": f"roadnet_{self.road_net}.json",
            "LIST_STATE_FEATURE": env_cfg.get(
                "list_state_feature",
                [
                    "cur_phase",
                    "traffic_movement_pressure_queue",
                ],
            ),
            "DIC_REWARD_INFO": env_cfg.get("dic_reward_info", {"queue_length": -0.25}),
        }

        self.dic_traffic_env_conf = copy.deepcopy(DIC_TRAFFIC_ENV_CONF)
        self.dic_traffic_env_conf.update(env_extra)

        self.run_counts = run_counts
        self.action_interval = action_interval

    def _prepare_paths(self) -> Dict[str, str]:
        paths_cfg = self.config["paths"]
        run_name = paths_cfg.get("run_name")
        if not run_name:
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            traffic_key = self.dic_traffic_env_conf["TRAFFIC_FILE"].replace(".json", "")
            run_name = f"qguided_grpo_{self.city_name.lower()}_{traffic_key}_{timestamp}"

        output_root = paths_cfg.get("output_root", "logs/rl_grpo")
        checkpoint_root = paths_cfg.get("checkpoint_root", "checkpoints/rl_grpo")
        data_root = paths_cfg.get("data_root", "data")

        path_to_work_directory = os.path.join(output_root, run_name)
        path_to_checkpoints = os.path.join(checkpoint_root, run_name)

        os.makedirs(path_to_work_directory, exist_ok=True)
        os.makedirs(path_to_checkpoints, exist_ok=True)

        return {
            "PATH_TO_WORK_DIRECTORY": path_to_work_directory,
            "PATH_TO_TRAINED_CHECKPOINTS": path_to_checkpoints,
            "PATH_TO_DATA": os.path.join(data_root, self.city_name),
        }

    def _copy_cityflow_files(self) -> None:
        roadnet = self.dic_traffic_env_conf["ROADNET_FILE"]
        traffic = self.dic_traffic_env_conf["TRAFFIC_FILE"]
        src_data = self.dic_path["PATH_TO_DATA"]
        dst_dir = self.dic_path["PATH_TO_WORK_DIRECTORY"]

        for file_name in (roadnet, traffic):
            src = os.path.join(src_data, file_name)
            dst = os.path.join(dst_dir, file_name)
            if not os.path.exists(src):
                raise FileNotFoundError(f"CityFlow data file not found: {src}")
            if not os.path.exists(dst):
                shutil.copy2(src, dst)

    def _init_env(self):
        from src.env.cityflow_env import CityFlowEnv

        env = CityFlowEnv(
            path_to_log=self.dic_path["PATH_TO_WORK_DIRECTORY"],
            path_to_work_directory=self.dic_path["PATH_TO_WORK_DIRECTORY"],
            dic_traffic_env_conf=self.dic_traffic_env_conf,
            dic_path=self.dic_path,
        )
        env.reset() # TODO: env resetting twice?
        return env

    def _init_policy_model(self):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers and torch are required for policy loading. "
                "Install dependencies from requirements-rl.txt."
            ) from exc

        policy_cfg = self.config["policy"]
        llm_path = policy_cfg["llm_path"]

        dtype_name = str(policy_cfg.get("dtype", "bfloat16")).lower()
        dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16

        model = AutoModelForCausalLM.from_pretrained(
            llm_path,
            dtype=dtype,
            device_map=policy_cfg.get("device_map", "auto"),
        )

        tokenizer = AutoTokenizer.from_pretrained(llm_path, padding_side="left")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

        lora_cfg = policy_cfg.get("lora", {})
        if lora_cfg.get("enabled", False):
            try:
                from peft import LoraConfig, get_peft_model
            except ImportError as exc:
                raise ImportError(
                    "peft is required for LoRA policy setup. Install dependencies from requirements-rl.txt."
                ) from exc
            model = get_peft_model(
                model,
                LoraConfig(
                    r=int(lora_cfg.get("r", 8)),
                    lora_alpha=int(lora_cfg.get("lora_alpha", 16)),
                    lora_dropout=float(lora_cfg.get("lora_dropout", 0.05)),
                    bias=lora_cfg.get("bias", "none"),
                    task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
                ),
            )

        return model, tokenizer

    def _init_q_rewarder(self):
        from src.rl.critic.colight_q_rewarder import CoLightQRewarder, CriticLoadSpec

        critic_cfg = self.config["critic"]
        checkpoint_file = critic_cfg.get("checkpoint_file")
        load_spec = None
        if checkpoint_file:
            load_spec = CriticLoadSpec(
                checkpoint_file=checkpoint_file,
                checkpoint_dir=critic_cfg.get("checkpoint_dir"),
            )

        critic_agent_conf = dict(DIC_BASE_AGENT_CONF)
        if "learning_rate" in critic_cfg:
            critic_agent_conf["LEARNING_RATE"] = float(critic_cfg["learning_rate"])

        return CoLightQRewarder(
            dic_agent_conf=critic_agent_conf,
            dic_traffic_env_conf=self.dic_traffic_env_conf,
            dic_path=self.dic_path,
            list_state_features=critic_cfg.get("list_state_feature"),
            cnn_layers=critic_cfg.get("cnn_layers", [[32, 32]]),
            load_spec=load_spec,
        )

    def _init_collector(self) -> QGuidedRolloutCollector:
        rollout_cfg = self.config["rollout"]
        return QGuidedRolloutCollector(
            env=self.env,
            tokenizer=self.tokenizer,
            model=self.policy_model,
            q_rewarder=self.q_rewarder,
            k=int(rollout_cfg.get("k", 4)),
            invalid_action_reward=float(rollout_cfg.get("invalid_action_reward", 0.0)),
            max_prompt_length=int(rollout_cfg.get("max_prompt_length", 2048)),
            max_new_tokens=int(rollout_cfg.get("max_new_tokens", 256)),
            temperature=float(rollout_cfg.get("temperature", 1.0)),
            top_p=float(rollout_cfg.get("top_p", 1.0)),
            top_k=int(rollout_cfg.get("top_k", 50)),
            do_sample=bool(rollout_cfg.get("do_sample", True)),
        )

    def _init_grpo_runner(self) -> GRPOTrainingRunner:
        grpo_cfg = copy.deepcopy(self.config["grpo"])
        grpo_cfg["num_generations"] = int(self.config["rollout"]["k"])
        grpo_cfg.setdefault("max_completion_length", int(self.config["rollout"].get("max_new_tokens", 256)))
        return GRPOTrainingRunner(
            grpo_config=grpo_cfg,
            invalid_action_reward=float(self.config["rollout"].get("invalid_action_reward", 0.0)),
        )

    def _save_policy_checkpoint(self, episode_idx: int) -> str:
        checkpoint_dir = os.path.join(
            self.dic_path["PATH_TO_TRAINED_CHECKPOINTS"],
            f"episode_{episode_idx}",
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        if hasattr(self.policy_model, "save_pretrained"):
            self.policy_model.save_pretrained(checkpoint_dir)
        if self.tokenizer is not None and hasattr(self.tokenizer, "save_pretrained"):
            self.tokenizer.save_pretrained(checkpoint_dir)
        return checkpoint_dir

    def run(self) -> List[Dict[str, Any]]:
        episodes = int(self.config["train"]["episodes"])
        save_every = int(self.config["train"].get("save_every_episode", 1))

        run_logs: List[Dict[str, Any]] = []
        for episode_idx in range(episodes):
            # rollout_result = self.collector.collect_episode(
            #     run_counts=self.run_counts,
            #     action_interval=self.action_interval,
            # )
            # trainer = self.grpo_runner.train_on_records(
            #     model=self.policy_model,
            #     tokenizer=self.tokenizer,
            #     records=rollout_result.dataset_records,
            # )
            import pickle
            with open('data.pkl', "rb") as f:
                dataset_records = pickle.load(f)

            trainer = self.grpo_runner.train_on_records(
                model=self.policy_model,
                tokenizer=self.tokenizer,
                records=dataset_records
            )

            checkpoint_path = None
            if save_every > 0 and (episode_idx + 1) % save_every == 0:
                checkpoint_path = self._save_policy_checkpoint(episode_idx)

            run_logs.append(
                {
                    "episode": episode_idx,
                    "steps": rollout_result.num_steps,
                    "dataset_records": len(rollout_result.dataset_records),
                    "checkpoint_path": checkpoint_path,
                    "trainer_built": trainer is not None,
                }
            )

        return run_logs

    def evaluate(self, episodes: int = 1) -> List[RolloutResult]:
        results: List[RolloutResult] = []
        for _ in range(int(episodes)):
            results.append(
                self.collector.collect_episode(
                    run_counts=self.run_counts,
                    action_interval=self.action_interval,
                )
            )
        return results
