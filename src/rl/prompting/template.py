from .parser import ACTION_TO_ID

_PHASE_EXPLANATION = {
    "NTST": "Northern and southern through lanes.",
    "NLSL": "Northern and southern left-turn lanes.",
    "ETWT": "Eastern and western through lanes.",
    "ELWL": "Eastern and western left-turn lanes.",
}


def build_prompt(state_text: str) -> str: # TODO: correct prompt
    """Build an instruction-style prompt for phase selection with reasoning."""
    signal_lines = "\n".join(f"- {phase}: {_PHASE_EXPLANATION[phase]}" for phase in ACTION_TO_ID)

    return (
        "You are an expert in traffic management.\n\n"
        "A traffic light controls a four-way intersection with one through lane and one left-turn lane per approach. "
        "Each lane has queued vehicles and approaching vehicles.\n\n"
        "Available signal phases:\n"
        f"{signal_lines}\n\n"
        "Current intersection state:\n"
        f"{state_text}\n\n"
        "Task:\n"
        "1) Reason briefly about which phase best improves near-term traffic conditions.\n"
        "2) Output exactly one final decision in this format: <signal>PHASE</signal>.\n"
        "3) PHASE must be one of: ETWT, NTST, ELWL, NLSL."
    )
