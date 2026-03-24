import re
from typing import Optional

ACTION_TO_ID = {
    "ETWT": 0,
    "NTST": 1,
    "ELWL": 2,
    "NLSL": 3,
}

ID_TO_ACTION = {v: k for k, v in ACTION_TO_ID.items()}
_SIGNAL_PATTERN = re.compile(r"<signal>(.*?)</signal>", re.IGNORECASE | re.DOTALL)


def parse_signal_text(completion: str) -> Optional[str]:
    """Extract the final <signal>...</signal> phase tag from a completion."""
    if not completion:
        return None
    matches = _SIGNAL_PATTERN.findall(completion)
    if not matches:
        return None
    signal = matches[-1].strip().upper()
    if signal not in ACTION_TO_ID:
        return None
    return signal


def parse_signal(completion: str) -> Optional[int]:
    """Return action id for the parsed signal, or None if invalid/missing."""
    signal = parse_signal_text(completion)
    if signal is None:
        return None
    return ACTION_TO_ID[signal]
