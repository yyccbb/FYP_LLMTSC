from .parser import ACTION_TO_ID

_PHASE_EXPLANATION = {
    "NTST": "Northern and southern through lanes.",
    "NLSL": "Northern and southern left-turn lanes.",
    "ETWT": "Eastern and western through lanes.",
    "ELWL": "Eastern and western left-turn lanes.",
}

def build_prompt(state_text: str) -> str:
    """Build an instruction-style prompt for traffic signal selection."""
    signal_lines = "\n".join(f"- {phase}: {_PHASE_EXPLANATION[phase]}" for phase in ACTION_TO_ID)
    allowed_phases = ", ".join(ACTION_TO_ID)

    return (
        "You are an expert in traffic management.\n\n"
        "A traffic light regulates a four-way intersection with northern, southern, eastern, and western approaches, "
        "each containing two lanes: one for through traffic and one for left-turns. Each lane is further divided into "
        "three segments. Segment 1 is the closest to the intersection. Segment 2 is in the middle. Segment 3 is the "
        "farthest. In a lane, there may be early queued vehicles and approaching vehicles traveling in different "
        "segments. Early queued vehicles have already arrived at the intersection and await passage permission. "
        "Approaching vehicles will arrive at the intersection in the future.\n\n"
        "The traffic light has 4 signal phases. Each signal relieves vehicles' flow in a group of two specific lanes.\n\n"
        "Available signal phases:\n"
        f"{signal_lines}\n\n"
        "Current intersection state:\n"
        f"{state_text}\n\n"
        "The state description above lists:\n"
        "- The group of lanes relieved under each traffic light phase.\n"
        "- The number of early queued vehicles in the allowed lanes of each signal.\n"
        "- The number of approaching vehicles in different segments of the allowed lanes of each signal.\n\n"
        "Question:\n"
        "Which is the most effective traffic signal that will most significantly improve the traffic condition during "
        "the next phase?\n\n"
        "Note:\n"
        "- Traffic congestion is primarily dictated by early queued vehicles, with the most significant impact.\n"
        "- You must pay the most attention to lanes with long queue lengths.\n"
        "- It is not urgent to consider vehicles in distant segments, since they are unlikely to reach the "
        "intersection soon.\n\n"
        "Requirements:\n"
        "- Think step by step.\n"
        "- You can only choose one of the signals listed above.\n"
        "- Step 1: Provide a brief analysis identifying the optimal traffic signal.\n"
        "- Step 2: After finishing the analysis, answer with your chosen signal.\n"
        f"- Include exactly one final decision tag in this format: <signal>PHASE</signal>, where PHASE is one of: {allowed_phases}."
    )
