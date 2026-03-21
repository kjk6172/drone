from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

from communication.send_command import send_command
from decision.action_mapper import map_action_to_command
from decision.state_builder import build_state
from llm.llm_inference import decide_action
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


def _run_follow_loop(
    user_command: str,
    obstacle: bool,
    send: bool,
    mock_person: Dict[str, float] | None,
    capture: Any,
    follow_steps: int,
    follow_interval_s: float,
) -> list[Dict[str, Any]]:
    trace: list[Dict[str, Any]] = []

    for step in range(follow_steps):
        person_result = _resolve_person_result(mock_person=mock_person, capture=capture)
        follow_state = _build_state_from_inputs(
            user_command=user_command,
            person_result=person_result,
            obstacle=obstacle,
        )
        follow_command = map_action_to_command("follow_person", follow_state)

        if send:
            send_command(follow_command)

        trace.append(
            {
                "step": step + 1,
                "state": follow_state,
                "drone_command": follow_command,
            }
        )

        if step < follow_steps - 1:
            time.sleep(follow_interval_s)

    return trace


def run_pipeline(
    user_command: str,
    mock_person: Dict[str, float] | None = None,
    obstacle: bool = False,
    send: bool = False,
    use_camera: bool = False,
    camera_index: int | None = None,
    follow_steps: int | None = None,
    follow_interval_s: float | None = None,
) -> Dict[str, Any]:
    config = _load_runtime_config()["vision"]
    resolved_camera_index = config["camera_index"] if camera_index is None else camera_index
    resolved_follow_steps = config["follow_loop_steps"] if follow_steps is None else follow_steps
    resolved_follow_interval = config["follow_loop_interval_s"] if follow_interval_s is None else follow_interval_s

    capture = create_video_capture(resolved_camera_index) if use_camera else None

    try:
        person_result = _resolve_person_result(mock_person=mock_person, capture=capture)
        state = _build_state_from_inputs(
            user_command=user_command,
            person_result=person_result,
            obstacle=obstacle,
        )
        llm_output = decide_action(state)
        drone_command = map_action_to_command(llm_output["action"], state)

        result = {
            "state": state,
            "llm_output": llm_output,
            "drone_command": drone_command,
        }

        if llm_output["action"] == "follow_person" and resolved_follow_steps > 0:
            follow_trace = _run_follow_loop(
                user_command=user_command,
                obstacle=obstacle,
                send=send,
                mock_person=mock_person,
                capture=capture,
                follow_steps=resolved_follow_steps,
                follow_interval_s=resolved_follow_interval,
            )
            result["follow_trace"] = follow_trace
            if follow_trace:
                result["state"] = follow_trace[-1]["state"]
                result["drone_command"] = follow_trace[-1]["drone_command"]
        elif send:
            send_command(drone_command)

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
        "--follow-steps",
        type=int,
        default=config["follow_loop_steps"],
        help="Number of follow-control cycles to run after the LLM selects follow_person.",
    )
    parser.add_argument(
        "--follow-interval",
        type=float,
        default=config["follow_loop_interval_s"],
        help="Delay in seconds between follow-control cycles.",
    )
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
        follow_steps=args.follow_steps,
        follow_interval_s=args.follow_interval,
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
