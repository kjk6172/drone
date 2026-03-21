from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "parameters.json"


def _load_parameters() -> Dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def _base_velocity_command() -> Dict[str, float]:
    config = _load_parameters()["control"]
    return {
        "vx": 0.0,
        "vy": 0.0,
        "vz": 0.0,
        "yaw_rate": 0.0,
        "duration": config["command_duration_s"],
    }


def map_action_to_command(action: str, state: Dict[str, Any]) -> Dict[str, Any]:
    config = _load_parameters()
    control = config["control"]
    drone = config["drone"]
    command = _base_velocity_command()

    if action == "takeoff":
        return {
            "command": "takeoff",
            "altitude_m": drone["takeoff_altitude_m"],
        }

    if action == "land":
        return {"command": "land"}

    if action == "hover":
        return {"command": "hover", **command}

    if action == "move_forward":
        return {"command": "velocity", **command, "vx": control["linear_speed_mps"]}

    if action == "move_backward":
        return {"command": "velocity", **command, "vx": -control["linear_speed_mps"]}

    if action == "turn_left":
        return {"command": "velocity", **command, "yaw_rate": -control["yaw_rate_degps"]}

    if action == "turn_right":
        return {"command": "velocity", **command, "yaw_rate": control["yaw_rate_degps"]}

    if action == "follow_person":
        if not state.get("person_detected"):
            return {"command": "hover", **command}

        person_position = state.get("person_position", "center")
        distance = state.get("distance", "unknown")
        follow_command = {"command": "velocity", **command}

        if person_position == "left":
            follow_command["yaw_rate"] = -control["yaw_rate_degps"]
        elif person_position == "right":
            follow_command["yaw_rate"] = control["yaw_rate_degps"]

        if distance == "far":
            follow_command["vx"] = control["linear_speed_mps"]
        elif distance == "near":
            follow_command["vx"] = 0.0

        return follow_command

    return {"command": "hover", **command}
