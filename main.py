from __future__ import annotations

import argparse
import json
from typing import Any, Dict

from communication.send_command import send_command
from decision.action_mapper import map_action_to_command
from decision.state_builder import build_state
from llm.llm_inference import decide_action
from vision.hand_gesture import detect_hand_gesture
from vision.person_detection import detect_person


def run_pipeline(
    user_command: str,
    mock_person: Dict[str, float] | None = None,
    obstacle: bool = False,
    send: bool = False,
) -> Dict[str, Any]:
    gesture_result = detect_hand_gesture()
    person_result = detect_person(mock_detection=mock_person)
    state = build_state(
        user_command=user_command,
        gesture_result=gesture_result,
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

    if send:
        send_command(drone_command)

    return result


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the AI drone decision pipeline.")
    parser.add_argument("user_command", help="Natural-language command for the drone.")
    parser.add_argument("--person", action="store_true", help="Simulate a detected person.")
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
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
