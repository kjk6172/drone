from __future__ import annotations

from typing import Any, Dict


def detect_hand_gesture(frame: Any = None) -> Dict[str, Any]:
    return {
        "gesture": "none",
        "confidence": 0.0,
        "hand_detected": False,
    }
