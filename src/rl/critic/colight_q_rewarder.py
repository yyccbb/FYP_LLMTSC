from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from src.modelling.agent.colight_agent import CoLightAgentTorch
from src.utils.config import DIC_BASE_AGENT_CONF


@dataclass
class CriticLoadSpec:
    checkpoint_file: str
    checkpoint_dir: Optional[str] = None


class CoLightQRewarder:
    """Thin wrapper around CoLight to provide per-action Q-values as rewards."""

    def __init__(
        self,
        dic_agent_conf: Dict[str, Any],
        dic_traffic_env_conf: Dict[str, Any],
        dic_path: Dict[str, str],
        list_state_features: Optional[List[str]] = None,
        cnn_layers: Optional[List[List[int]]] = None,
        load_spec: Optional[CriticLoadSpec] = None,
    ) -> None:
        self.dic_path = dic_path
        self.list_state_features = list_state_features or [
            "cur_phase",
            "traffic_movement_pressure_queue_efficient",
            "lane_enter_running_part",
            "adjacency_matrix",
        ]

        critic_agent_conf = dict(DIC_BASE_AGENT_CONF)
        critic_agent_conf.update(dic_agent_conf)
        critic_agent_conf.update(
            {
                "EPSILON": 0.0,
                "MIN_EPSILON": 0.0,
                "CNN_layers": cnn_layers or [[32, 32]],
            }
        )

        critic_env_conf = dict(dic_traffic_env_conf)
        critic_env_conf["LIST_STATE_FEATURE"] = self.list_state_features

        self.agent = CoLightAgentTorch(
            dic_agent_conf=critic_agent_conf,
            dic_traffic_env_conf=critic_env_conf,
            dic_path=dic_path,
            cnt_round=0,
            intersection_id="0",
        )

        if load_spec is not None:
            self.agent.load_network(load_spec.checkpoint_file, file_path=load_spec.checkpoint_dir)

    def get_q_values(self, state) -> np.ndarray:
        """Return normalized CoLight values with shape [num_intersections, num_actions]."""
        _, q_values = self.agent.choose_action_with_value(0, state)
        return q_values
