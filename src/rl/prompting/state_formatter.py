from copy import deepcopy
from typing import Dict, List

from src.utils.my_utils import get_state_detail

from .template import build_prompt

_LOCATION_DICT = {"N": "North", "S": "South", "E": "East", "W": "West"}
_PHASES = ["ETWT", "NTST", "ELWL", "NLSL"]
_PHASE_EXPLANATION = {
    "NTST": "Northern and southern through lanes.",
    "NLSL": "Northern and southern left-turn lanes.",
    "ETWT": "Eastern and western through lanes.",
    "ELWL": "Eastern and western left-turn lanes.",
}


def format_state_to_text(state: Dict[str, Dict[str, List[float]]]) -> str:
    """Convert structured lane state into a deterministic plain-text description."""
    lines = []
    for phase in _PHASES:
        lane_1 = phase[:2]
        lane_2 = phase[2:]

        queue_1 = int(state[lane_1]["queue_len"])
        queue_2 = int(state[lane_2]["queue_len"])

        seg_1_lane_1 = int(state[lane_1]["cells"][0])
        seg_2_lane_1 = int(state[lane_1]["cells"][1])
        seg_3_lane_1 = int(state[lane_1]["cells"][2] + state[lane_1]["cells"][3])

        seg_1_lane_2 = int(state[lane_2]["cells"][0])
        seg_2_lane_2 = int(state[lane_2]["cells"][1])
        seg_3_lane_2 = int(state[lane_2]["cells"][2] + state[lane_2]["cells"][3])

        lines.extend(
            [
                f"Signal: {phase}",
                f"Relieves: {_PHASE_EXPLANATION[phase]}",
                (
                    "- Queue: "
                    f"{queue_1} ({_LOCATION_DICT[lane_1[0]]}), "
                    f"{queue_2} ({_LOCATION_DICT[lane_2[0]]}), "
                    f"{queue_1 + queue_2} (Total)"
                ),
                (
                    "- Segment 1: "
                    f"{seg_1_lane_1} ({_LOCATION_DICT[lane_1[0]]}), "
                    f"{seg_1_lane_2} ({_LOCATION_DICT[lane_2[0]]}), "
                    f"{seg_1_lane_1 + seg_1_lane_2} (Total)"
                ),
                (
                    "- Segment 2: "
                    f"{seg_2_lane_1} ({_LOCATION_DICT[lane_1[0]]}), "
                    f"{seg_2_lane_2} ({_LOCATION_DICT[lane_2[0]]}), "
                    f"{seg_2_lane_1 + seg_2_lane_2} (Total)"
                ),
                (
                    "- Segment 3: "
                    f"{seg_3_lane_1} ({_LOCATION_DICT[lane_1[0]]}), "
                    f"{seg_3_lane_2} ({_LOCATION_DICT[lane_2[0]]}), "
                    f"{seg_3_lane_1 + seg_3_lane_2} (Total)"
                ),
                "",
            ]
        )
    return "\n".join(lines).strip()


def get_intersection_state(env, intersection_index: int):
    """Extract the prompt state for one intersection from CityFlowEnv."""
    inter_name = env.list_intersection[intersection_index].inter_name
    intersection = env.intersection_dict[inter_name]
    roads = deepcopy(intersection["roads"])
    statistic_state, _, _ = get_state_detail(roads, env)
    return statistic_state


def build_prompts_from_env(env) -> List[str]:
    """Build one prompt per intersection for the current environment state."""
    prompts: List[str] = []
    for i in range(len(env.list_intersection)):
        statistic_state = get_intersection_state(env, i)
        state_text = format_state_to_text(statistic_state)
        prompts.append(build_prompt(state_text))
    return prompts
