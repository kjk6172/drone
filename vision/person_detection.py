from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import cv2
except ImportError:  # pragma: no cover - optional runtime dependency
    cv2 = None


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "parameters.json"
_HOG_DETECTOR = None


def _load_vision_config() -> Dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = json.load(file)
    return config["vision"]


def infer_position(x_center_ratio: float) -> str:
    config = _load_vision_config()
    if x_center_ratio < config["left_boundary_ratio"]:
        return "left"
    if x_center_ratio < config["right_boundary_ratio"]:
        return "center"
    return "right"


def infer_distance(box_width_ratio: float) -> str:
    config = _load_vision_config()
    if box_width_ratio >= config["near_width_ratio"]:
        return "near"
    return "far"


def _default_result(source: str) -> Dict[str, Any]:
    return {
        "person_detected": False,
        "person_position": "none",
        "distance": "unknown",
        "source": source,
    }


def _get_hog_detector():
    global _HOG_DETECTOR
    if cv2 is None:
        raise RuntimeError("opencv-python is not installed.")
    if _HOG_DETECTOR is None:
        detector = cv2.HOGDescriptor()
        detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        _HOG_DETECTOR = detector
    return _HOG_DETECTOR


def _result_from_ratios(
    x_center_ratio: float,
    box_width_ratio: float,
    confidence: float = 1.0,
    bbox: Optional[list[int]] = None,
    source: str = "mock",
) -> Dict[str, Any]:
    result = {
        "person_detected": True,
        "person_position": infer_position(x_center_ratio),
        "distance": infer_distance(box_width_ratio),
        "x_center_ratio": round(x_center_ratio, 3),
        "box_width_ratio": round(box_width_ratio, 3),
        "person_confidence": round(confidence, 3),
        "source": source,
    }
    if bbox is not None:
        result["bbox"] = bbox
    return result


def _select_largest_detection(boxes: Any, weights: Any) -> tuple[Optional[tuple[int, int, int, int]], float]:
    best_box = None
    best_score = -1.0

    for index, box in enumerate(boxes):
        x, y, width, height = box
        area = width * height
        weight = float(weights[index]) if len(weights) > index else 1.0
        score = area * max(weight, 0.01)
        if score > best_score:
            best_box = (int(x), int(y), int(width), int(height))
            best_score = weight

    return best_box, best_score


def create_video_capture(camera_index: int = 0):
    if cv2 is None:
        raise RuntimeError("opencv-python is not installed.")

    vision = _load_vision_config()
    backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0
    capture = cv2.VideoCapture(camera_index, backend)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, vision["frame_width"])
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, vision["frame_height"])

    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Unable to open camera index {camera_index}.")

    return capture


def read_person_from_capture(capture: Any) -> Dict[str, Any]:
    success, frame = capture.read()
    if not success or frame is None:
        return _default_result(source="camera")
    return detect_person(frame=frame)


def detect_person(
    frame: Any = None,
    mock_detection: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    if mock_detection:
        return _result_from_ratios(
            x_center_ratio=mock_detection["x_center_ratio"],
            box_width_ratio=mock_detection["box_width_ratio"],
            source="mock",
        )

    if frame is None:
        return _default_result(source="none")

    if cv2 is None:
        return _default_result(source="opencv_unavailable")

    detector = _get_hog_detector()
    boxes, weights = detector.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8), scale=1.05)

    if len(boxes) == 0:
        return _default_result(source="camera")

    best_box, confidence = _select_largest_detection(boxes, weights)
    if best_box is None:
        return _default_result(source="camera")

    x, y, width, height = best_box
    frame_height, frame_width = frame.shape[:2]
    x_center_ratio = (x + (width / 2)) / frame_width
    box_width_ratio = width / frame_width

    return _result_from_ratios(
        x_center_ratio=x_center_ratio,
        box_width_ratio=box_width_ratio,
        confidence=confidence,
        bbox=[x, y, width, height],
        source="camera",
    )
