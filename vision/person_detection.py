from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import cv2
except ImportError:  # pragma: no cover - optional runtime dependency
    cv2 = None

try:
    from ultralytics import YOLO  # type: ignore
except ImportError:  # pragma: no cover - optional runtime dependency
    YOLO = None


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "parameters.json"
_HOG_DETECTOR = None
_YOLO_MODEL = None
_YOLO_MODEL_NAME = None
_TRACK_STATE: Dict[str, Any] = {
    "bbox": None,
    "confidence": 0.0,
    "missed_frames": 0,
}


def _load_vision_config() -> Dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = json.load(file)
    return config["vision"]


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


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


def _get_yolo_model(model_name: str):
    global _YOLO_MODEL, _YOLO_MODEL_NAME

    if YOLO is None:
        raise RuntimeError("ultralytics is not installed.")

    if _YOLO_MODEL is None or _YOLO_MODEL_NAME != model_name:
        _YOLO_MODEL = YOLO(model_name)
        _YOLO_MODEL_NAME = model_name

    return _YOLO_MODEL


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


def _bbox_to_ratios(bbox: tuple[int, int, int, int], frame_width: int) -> tuple[float, float]:
    x, _, width, _ = bbox
    x_center_ratio = (x + (width / 2.0)) / float(frame_width)
    box_width_ratio = width / float(frame_width)
    return x_center_ratio, box_width_ratio


def _bbox_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    ax2 = ax + aw
    ay2 = ay + ah
    bx2 = bx + bw
    by2 = by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = aw * ah
    area_b = bw * bh
    union_area = area_a + area_b - inter_area
    if union_area <= 0:
        return 0.0

    return inter_area / union_area


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


def _choose_detection_with_tracking(
    detections: list[tuple[tuple[int, int, int, int], float]],
    previous_bbox: Optional[tuple[int, int, int, int]],
) -> tuple[Optional[tuple[int, int, int, int]], float]:
    if not detections:
        return None, 0.0

    best_box = None
    best_conf = 0.0
    best_score = -1.0

    for bbox, conf in detections:
        area = bbox[2] * bbox[3]
        if previous_bbox is None:
            score = area * max(conf, 0.01)
        else:
            iou = _bbox_iou(bbox, previous_bbox)
            score = (2.0 * iou) + (0.5 * conf) + (area / 100000.0)

        if score > best_score:
            best_score = score
            best_box = bbox
            best_conf = conf

    return best_box, best_conf


def _extract_yolo_detections(frame: Any, vision_config: Dict[str, Any]) -> list[tuple[tuple[int, int, int, int], float]]:
    model_name = str(vision_config.get("yolo_model", "yolov8n.pt"))
    conf_th = float(vision_config.get("yolo_conf_threshold", 0.35))
    iou_th = float(vision_config.get("yolo_iou_threshold", 0.45))

    model = _get_yolo_model(model_name)
    results = model.predict(
        source=frame,
        classes=[0],  # person class
        conf=conf_th,
        iou=iou_th,
        verbose=False,
    )

    detections: list[tuple[tuple[int, int, int, int], float]] = []
    if not results:
        return detections

    result = results[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return detections

    for box in boxes:
        xyxy = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        x1, y1, x2, y2 = xyxy

        x = int(round(x1))
        y = int(round(y1))
        width = max(1, int(round(x2 - x1)))
        height = max(1, int(round(y2 - y1)))
        detections.append(((x, y, width, height), conf))

    return detections


def _update_track_state(bbox: tuple[int, int, int, int], confidence: float) -> None:
    _TRACK_STATE["bbox"] = bbox
    _TRACK_STATE["confidence"] = confidence
    _TRACK_STATE["missed_frames"] = 0


def _advance_track_state() -> None:
    _TRACK_STATE["missed_frames"] = int(_TRACK_STATE.get("missed_frames", 0)) + 1


def _reset_track_state() -> None:
    _TRACK_STATE["bbox"] = None
    _TRACK_STATE["confidence"] = 0.0
    _TRACK_STATE["missed_frames"] = 0


def _detect_person_with_yolo(frame: Any, vision_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    frame_height, frame_width = frame.shape[:2]

    try:
        detections = _extract_yolo_detections(frame, vision_config)
    except Exception:
        return None

    previous_bbox = _TRACK_STATE.get("bbox")
    if previous_bbox is not None and isinstance(previous_bbox, list):
        previous_bbox = tuple(int(v) for v in previous_bbox)

    best_box, confidence = _choose_detection_with_tracking(detections, previous_bbox)
    if best_box is not None:
        _update_track_state(best_box, confidence)
        x_center_ratio, box_width_ratio = _bbox_to_ratios(best_box, frame_width)
        return _result_from_ratios(
            x_center_ratio=x_center_ratio,
            box_width_ratio=box_width_ratio,
            confidence=confidence,
            bbox=[best_box[0], best_box[1], best_box[2], best_box[3]],
            source="camera_yolo",
        )

    persist_frames = int(vision_config.get("track_persist_frames", 2))
    allow_last_known = bool(vision_config.get("track_enable_last_known", True))

    if allow_last_known and previous_bbox is not None:
        _advance_track_state()
        if int(_TRACK_STATE.get("missed_frames", 0)) <= persist_frames:
            x_center_ratio, box_width_ratio = _bbox_to_ratios(previous_bbox, frame_width)
            track_confidence = _clamp(float(_TRACK_STATE.get("confidence", 0.0)) * 0.85, 0.05, 1.0)
            return _result_from_ratios(
                x_center_ratio=x_center_ratio,
                box_width_ratio=box_width_ratio,
                confidence=track_confidence,
                bbox=[previous_bbox[0], previous_bbox[1], previous_bbox[2], previous_bbox[3]],
                source="camera_yolo_track",
            )

    _reset_track_state()
    return _default_result(source="camera_yolo")


def _detect_person_with_hog(frame: Any) -> Dict[str, Any]:
    detector = _get_hog_detector()
    boxes, weights = detector.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8), scale=1.05)

    if len(boxes) == 0:
        return _default_result(source="camera_hog")

    best_box, confidence = _select_largest_detection(boxes, weights)
    if best_box is None:
        return _default_result(source="camera_hog")

    x, y, width, height = best_box
    frame_height, frame_width = frame.shape[:2]
    x_center_ratio = (x + (width / 2)) / frame_width
    box_width_ratio = width / frame_width

    return _result_from_ratios(
        x_center_ratio=x_center_ratio,
        box_width_ratio=box_width_ratio,
        confidence=confidence,
        bbox=[x, y, width, height],
        source="camera_hog",
    )


def create_video_capture(camera_index: int = 0, stream_url: str | None = None):
    if cv2 is None:
        raise RuntimeError("opencv-python is not installed.")

    vision = _load_vision_config()
    if stream_url:
        capture = cv2.VideoCapture(stream_url)
    else:
        backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0
        capture = cv2.VideoCapture(camera_index, backend)

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, vision["frame_width"])
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, vision["frame_height"])

    if not capture.isOpened():
        capture.release()
        source = stream_url if stream_url else f"camera index {camera_index}"
        raise RuntimeError(f"Unable to open video source: {source}.")

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

    vision_config = _load_vision_config()
    backend = str(vision_config.get("person_detector_backend", "hog")).strip().lower()

    if backend == "yolo":
        yolo_result = _detect_person_with_yolo(frame, vision_config)
        if yolo_result is not None and yolo_result.get("person_detected"):
            return yolo_result

        if not bool(vision_config.get("yolo_fallback_to_hog", True)):
            return yolo_result or _default_result(source="camera_yolo")

    return _detect_person_with_hog(frame)
