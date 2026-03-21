from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional


DEFAULT_HAND_STATE: Dict[str, Any] = {
    "gesture": "none",
    "confidence": 0.0,
    "hand_detected": False,
}

DEFAULT_PERSON_STATE: Dict[str, Any] = {
    "person_detected": False,
    "person_position": "none",
    "distance": "unknown",
}


def _merge_defaults(defaults: Dict[str, Any], payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    result = deepcopy(defaults)
    if payload:
        result.update(payload)
    return result


def build_state(
    user_command: str,
    gesture_result: Optional[Dict[str, Any]] = None,
    person_result: Optional[Dict[str, Any]] = None,
    obstacle: bool = False,
    extra_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    hand_state = _merge_defaults(DEFAULT_HAND_STATE, gesture_result)
    person_state = _merge_defaults(DEFAULT_PERSON_STATE, person_result)

    state = {
        "user_command": user_command.strip(),
        "gesture": hand_state["gesture"],
        "gesture_confidence": hand_state["confidence"],
        "hand_detected": hand_state["hand_detected"],
        "person_detected": person_state["person_detected"],
        "person_position": person_state["person_position"],
        "distance": person_state["distance"],
        "obstacle": bool(obstacle),
    }

    if extra_state:
        state.update(extra_state)

    return state
