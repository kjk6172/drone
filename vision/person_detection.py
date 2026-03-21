from __future__ import annotations

from typing import Any, Dict, Optional


def infer_position(x_center_ratio: float) -> str:
    if x_center_ratio < 1 / 3:
        return "left"
    if x_center_ratio < 2 / 3:
        return "center"
    return "right"


def infer_distance(box_width_ratio: float) -> str:
    if box_width_ratio >= 0.45:
        return "near"
    return "far"


def detect_person(
    frame: Any = None,
    mock_detection: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    if not mock_detection:
        return {
            "person_detected": False,
            "person_position": "none",
            "distance": "unknown",
        }

    return {
        "person_detected": True,
        "person_position": infer_position(mock_detection["x_center_ratio"]),
        "distance": infer_distance(mock_detection["box_width_ratio"]),
    }
