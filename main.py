from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

from communication.send_command import send_command
from decision.action_mapper import map_intent_to_command
from decision.state_builder import build_state
from llm.llm_inference import decide_intent
from vision.hand_gesture import detect_hand_gesture
from vision.person_detection import create_video_capture, detect_person, read_person_from_capture


CONFIG_PATH = Path(__file__).resolve().parent / "config" / "parameters.json"


def _load_runtime_config() -> Dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def _build_state_from_inputs(
    user_command: str,
    person_result: Dict[str, Any],
    obstacle: bool = False,
) -> Dict[str, Any]:
    gesture_result = detect_hand_gesture()
    return build_state(
        user_command=user_command,
        gesture_result=gesture_result,
        person_result=person_result,
        obstacle=obstacle,
    )


def _resolve_person_result(
    mock_person: Dict[str, float] | None,
    capture: Any = None,
) -> Dict[str, Any]:
    if mock_person is not None:
        return detect_person(mock_detection=mock_person)
    if capture is not None:
        return read_person_from_capture(capture)
    return detect_person()


def _should_run_realtime_loop(intent_output: Dict[str, Any]) -> bool:
    intent = intent_output.get("intent")
    params = intent_output.get("params", {})
    if not isinstance(params, dict):
        params = {}

    if intent == "track_target":
        return bool(params.get("continuous", True))
    return False


def _run_realtime_control_loop(
    user_command: str,
    intent_output: Dict[str, Any],
    obstacle: bool,
    send: bool,
    mock_person: Dict[str, float] | None,
    capture: Any,
    control_steps: int,
    control_interval_s: float,
    target_host: str | None = None,
    target_port: int | None = None,
) -> list[Dict[str, Any]]:
    trace: list[Dict[str, Any]] = []

    for step in range(control_steps):
        person_result = _resolve_person_result(mock_person=mock_person, capture=capture)
        loop_state = _build_state_from_inputs(
            user_command=user_command,
            person_result=person_result,
            obstacle=obstacle,
        )
        loop_command = map_intent_to_command(intent_output, loop_state)

        if send:
            send_command(loop_command, host=target_host, port=target_port)

        trace.append(
            {
                "step": step + 1,
                "state": loop_state,
                "drone_command": loop_command,
            }
        )

        if step < control_steps - 1:
            time.sleep(control_interval_s)

    return trace


def run_pipeline(
    user_command: str,
    mock_person: Dict[str, float] | None = None,
    obstacle: bool = False,
    send: bool = False,
    use_camera: bool = False,
    camera_index: int | None = None,
    stream_url: str | None = None,
    control_steps: int | None = None,
    control_interval_s: float | None = None,
    target_host: str | None = None,
    target_port: int | None = None,
) -> Dict[str, Any]:
    config = _load_runtime_config()["vision"]
    resolved_camera_index = config["camera_index"] if camera_index is None else camera_index
    resolved_control_steps = (
        config.get("realtime_loop_steps", config.get("follow_loop_steps", 10))
        if control_steps is None
        else control_steps
    )
    resolved_control_interval = (
        config.get("realtime_loop_interval_s", config.get("follow_loop_interval_s", 0.5))
        if control_interval_s is None
        else control_interval_s
    )

    capture = create_video_capture(resolved_camera_index, stream_url=stream_url) if use_camera else None

    try:
        person_result = _resolve_person_result(mock_person=mock_person, capture=capture)
        state = _build_state_from_inputs(
            user_command=user_command,
            person_result=person_result,
            obstacle=obstacle,
        )
        intent_output = decide_intent(state)
        drone_command = map_intent_to_command(intent_output, state)

        result = {
            "state": state,
            "intent_output": intent_output,
            "drone_command": drone_command,
        }

        if _should_run_realtime_loop(intent_output) and resolved_control_steps > 0:
            control_trace = _run_realtime_control_loop(
                user_command=user_command,
                intent_output=intent_output,
                obstacle=obstacle,
                send=send,
                mock_person=mock_person,
                capture=capture,
                control_steps=resolved_control_steps,
                control_interval_s=resolved_control_interval,
                target_host=target_host,
                target_port=target_port,
            )
            result["control_trace"] = control_trace
            if control_trace:
                result["state"] = control_trace[-1]["state"]
                result["drone_command"] = control_trace[-1]["drone_command"]
        elif send:
            send_command(drone_command, host=target_host, port=target_port)

        return result
    finally:
        if capture is not None:
            capture.release()


def _build_argument_parser() -> argparse.ArgumentParser:
    config = _load_runtime_config()["vision"]
    parser = argparse.ArgumentParser(description="Run the AI drone decision pipeline.")
    parser.add_argument("user_command", help="Natural-language command for the drone.")
    parser.add_argument("--person", action="store_true", help="Simulate a detected person.")
    parser.add_argument("--camera", action="store_true", help="Use a live camera frame for person detection.")
    parser.add_argument("--camera-index", type=int, default=config["camera_index"], help="Camera index for OpenCV capture.")
    parser.add_argument(
        "--stream-url",
        default=config.get("stream_url"),
        help="Optional RTSP/UDP/HTTP video stream URL. If set, it overrides --camera-index.",
    )
    parser.add_argument(
        "--position",
        choices=["left", "center", "right"],
        default="center",
        help="Simulated person position when --person is used.",
    )
    parser.add_argument(
        "--distance",
        choices=["near", "far"],
        default="far",
        help="Simulated person distance when --person is used.",
    )
    parser.add_argument("--obstacle", action="store_true", help="Simulate obstacle detection.")
    parser.add_argument("--send", action="store_true", help="Send the resulting command over socket.")
    parser.add_argument(
        "--target-host",
        default=None,
        help="Override command target host. Useful when sending from laptop to Raspberry Pi.",
    )
    parser.add_argument(
        "--target-port",
        type=int,
        default=None,
        help="Override command target port.",
    )

    default_steps = config.get("realtime_loop_steps", config.get("follow_loop_steps", 10))
    default_interval = config.get("realtime_loop_interval_s", config.get("follow_loop_interval_s", 0.5))

    parser.add_argument(
        "--control-steps",
        type=int,
        default=default_steps,
        help="Number of real-time control cycles for continuous intents like track_target.",
    )
    parser.add_argument(
        "--control-interval",
        type=float,
        default=default_interval,
        help="Delay in seconds between control cycles.",
    )

    # Backward-compatible aliases
    parser.add_argument("--follow-steps", type=int, dest="control_steps", help=argparse.SUPPRESS)
    parser.add_argument("--follow-interval", type=float, dest="control_interval", help=argparse.SUPPRESS)

    return parser


def _mock_person_payload(position: str, distance: str) -> Dict[str, float]:
    x_center_ratio = {"left": 0.15, "center": 0.5, "right": 0.85}[position]
    box_width_ratio = {"near": 0.55, "far": 0.2}[distance]
    return {
        "x_center_ratio": x_center_ratio,
        "box_width_ratio": box_width_ratio,
    }


if __name__ == "__main__":
    parser = _build_argument_parser()
    args = parser.parse_args()

    mock_person = None
    if args.person:
        mock_person = _mock_person_payload(args.position, args.distance)

    output = run_pipeline(
        user_command=args.user_command,
        mock_person=mock_person,
        obstacle=args.obstacle,
        send=args.send,
        use_camera=args.camera,
        camera_index=args.camera_index,
        stream_url=args.stream_url,
        control_steps=args.control_steps,
        control_interval_s=args.control_interval,
        target_host=args.target_host,
        target_port=args.target_port,
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
