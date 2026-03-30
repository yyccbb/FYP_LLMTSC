from copy import deepcopy
from typing import Dict, List, Optional, Tuple

from .template import build_prompt

_LOCATION_DICT = {"N": "North", "S": "South", "E": "East", "W": "West"}
_PHASES = ["ETWT", "NTST", "ELWL", "NLSL"]
_PHASE_EXPLANATION = {
    "NTST": "Northern and southern through lanes.",
    "NLSL": "Northern and southern left-turn lanes.",
    "ETWT": "Eastern and western through lanes.",
    "ELWL": "Eastern and western left-turn lanes.",
}
_DIRECTION_TO_ROAD_KEY = {"T": "go_straight", "L": "turn_left", "R": "turn_right"}
# neighbor_ENWS comes from road suffix order 0,1,2,3 -> W,S,E,N in this environment.
_SIDE_TO_NEIGHBOR_INDEX = {"W": 0, "S": 1, "E": 2, "N": 3}
_SIDE_TO_MOVEMENTS = {
    "W": ("NL", "WT", "SR"),
    "E": ("NR", "ET", "SL"),
    "N": ("EL", "NT", "WR"),
    "S": ("ER", "ST", "WL"),
}
_PHASE_TO_SIDES = {
    "ETWT": ("E", "W"),
    "ELWL": ("E", "W"),
    "NTST": ("N", "S"),
    "NLSL": ("N", "S"),
}


def _lookup_neighbor_id(env, intersection_index: int, side: str) -> Optional[str]:
    neighbor_ids = env.list_intersection[intersection_index].neighbor_ENWS
    return neighbor_ids[_SIDE_TO_NEIGHBOR_INDEX[side]]


def _count_neighbor_movement_vehicles(neighbor_roads: Dict[str, Dict], movement: str, lane_vehicles: Dict[str, List[str]]) -> int:
    location = _LOCATION_DICT[movement[0]]
    direction_key = _DIRECTION_TO_ROAD_KEY[movement[1]]
    total = 0

    for road_id, road_info in neighbor_roads.items():
        if road_info["type"] != "outgoing" or road_info["location"] != location:
            continue
        lane_ids = road_info["lanes"].get(direction_key, [])
        for lane_idx in lane_ids:
            total += len(lane_vehicles.get(f"{road_id}_{lane_idx}", []))
    return total


def _build_neighbor_side_totals(
    env,
    intersection_index: int,
    lane_vehicles: Dict[str, List[str]],
) -> Dict[str, Optional[int]]:
    side_totals: Dict[str, Optional[int]] = {}

    for side, movements in _SIDE_TO_MOVEMENTS.items():
        neighbor_id = _lookup_neighbor_id(env, intersection_index, side)
        if neighbor_id is None or neighbor_id not in env.intersection_dict:
            side_totals[side] = None
            continue

        neighbor_roads = env.intersection_dict[neighbor_id]["roads"]
        side_totals[side] = sum(
            _count_neighbor_movement_vehicles(neighbor_roads, movement, lane_vehicles)
            for movement in movements
        )

    return side_totals


def _format_neighbor_line(phase: str, side_totals: Dict[str, Optional[int]]) -> str:
    side_1, side_2 = _PHASE_TO_SIDES[phase]
    value_1 = side_totals[side_1]
    value_2 = side_totals[side_2]

    display_1 = "NA" if value_1 is None else str(value_1)
    display_2 = "NA" if value_2 is None else str(value_2)

    known_values = [value for value in (value_1, value_2) if value is not None]
    available = len(known_values)
    known_total = "NA" if available == 0 else str(sum(known_values))

    return (
        "- Neighbor incoming totals: "
        f"{display_1} ({_LOCATION_DICT[side_1]}), "
        f"{display_2} ({_LOCATION_DICT[side_2]}), "
        f"{known_total} (Known total), "
        f"{available}/2 available"
    )


def format_state_to_text(state: Dict[str, Dict[str, List[float]]], side_totals: Dict[str, Optional[int]]) -> str:
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
                    "- Early queued: "
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
                _format_neighbor_line(phase, side_totals),
                "",
            ]
        )
    return "\n".join(lines).strip()


def get_intersection_state(
    env,
    intersection_index: int,
    lane_vehicles: Dict[str, List[str]],
) -> Tuple[Dict[str, Dict[str, List[float]]], Dict[str, Optional[int]]]:
    """Extract the prompt state for one intersection from CityFlowEnv."""
    from src.utils.my_utils import get_state_detail

    inter_name = env.list_intersection[intersection_index].inter_name
    intersection = env.intersection_dict[inter_name]
    roads = deepcopy(intersection["roads"])
    statistic_state, _, _ = get_state_detail(roads, env)
    side_totals = _build_neighbor_side_totals(env, intersection_index, lane_vehicles)
    return statistic_state, side_totals


def build_prompts_from_env(env) -> List[str]:
    """Build one prompt per intersection for the current environment state."""
    prompts: List[str] = []
    lane_vehicles = env.eng.get_lane_vehicles()
    for i in range(len(env.list_intersection)):
        statistic_state, side_totals = get_intersection_state(env, i, lane_vehicles)
        state_text = format_state_to_text(statistic_state, side_totals)
        prompts.append(build_prompt(state_text))
    return prompts
