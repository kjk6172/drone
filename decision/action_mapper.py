from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "parameters.json"


def _load_parameters() -> Dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def _base_velocity_command(duration_s: float) -> Dict[str, float]:
    return {
        "vx": 0.0,
        "vy": 0.0,
        "vz": 0.0,
        "yaw_rate": 0.0,
        "duration": duration_s,
    }


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _read_float(payload: Dict[str, Any], key: str, default: float) -> float:
    value = payload.get(key, default)
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _safe_hover_command(duration_s: float) -> Dict[str, Any]:
    return {
        "command": "hover",
        **_base_velocity_command(duration_s),
    }


def _apply_safety_limits(command: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    if command.get("command") != "velocity":
        return command

    safety = config.get("safety", {})
    max_linear = float(safety.get("max_linear_speed_mps", config["control"]["linear_speed_mps"]))
    max_vertical = float(safety.get("max_vertical_speed_mps", config["control"]["vertical_speed_mps"]))
    max_yaw = float(safety.get("max_yaw_rate_degps", config["control"]["yaw_rate_degps"]))
    min_duration = float(safety.get("min_command_duration_s", 0.1))
    max_duration = float(safety.get("max_command_duration_s", 2.0))

    safe = dict(command)
    safe["vx"] = round(_clamp(float(safe.get("vx", 0.0)), -max_linear, max_linear), 3)
    safe["vy"] = round(_clamp(float(safe.get("vy", 0.0)), -max_linear, max_linear), 3)
    safe["vz"] = round(_clamp(float(safe.get("vz", 0.0)), -max_vertical, max_vertical), 3)
    safe["yaw_rate"] = round(_clamp(float(safe.get("yaw_rate", 0.0)), -max_yaw, max_yaw), 3)
    safe["duration"] = round(_clamp(float(safe.get("duration", min_duration)), min_duration, max_duration), 3)
    return safe


def _move_relative_command(intent_params: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    control = config["control"]
    duration = _read_float(intent_params, "duration_s", control["command_duration_s"])
    command = {
        "command": "velocity",
        **_base_velocity_command(duration),
    }

    forward = _clamp(_read_float(intent_params, "forward", 0.0), -1.0, 1.0)
    right = _clamp(_read_float(intent_params, "right", 0.0), -1.0, 1.0)
    up = _clamp(_read_float(intent_params, "up", 0.0), -1.0, 1.0)
    yaw = _clamp(_read_float(intent_params, "yaw", 0.0), -1.0, 1.0)

    command["vx"] = forward * float(control["linear_speed_mps"])
    command["vy"] = right * float(control["linear_speed_mps"])
    command["vz"] = up * float(control["vertical_speed_mps"])
    command["yaw_rate"] = yaw * float(control["yaw_rate_degps"])
    return command


def _track_target_command(intent_params: Dict[str, Any], state: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    control = config["control"]
    duration = _read_float(intent_params, "duration_s", control["command_duration_s"])

    if not state.get("person_detected"):
        return _safe_hover_command(duration)

    command = {
        "command": "velocity",
        **_base_velocity_command(duration),
    }

    x_center_ratio = state.get("x_center_ratio")
    deadzone = float(control.get("track_target_deadzone_ratio", 0.08))
    yaw_gain = float(control.get("track_target_yaw_gain", 1.0))

    if isinstance(x_center_ratio, (int, float)):
        x_error = float(x_center_ratio) - 0.5
        if abs(x_error) > deadzone:
            normalized_error = _clamp(x_error / 0.5, -1.0, 1.0)
            command["yaw_rate"] = normalized_error * float(control["yaw_rate_degps"]) * yaw_gain
    else:
        person_position = state.get("person_position", "center")
        if person_position == "left":
            command["yaw_rate"] = -float(control["yaw_rate_degps"])
        elif person_position == "right":
            command["yaw_rate"] = float(control["yaw_rate_degps"])

    desired_box_width_ratio = _clamp(
        _read_float(intent_params, "desired_box_width_ratio", float(control["track_target_box_width_ratio"])),
        0.1,
        0.8,
    )
    forward_bias = _clamp(_read_float(intent_params, "forward_bias", 0.0), -1.0, 1.0)

    box_width_ratio = state.get("box_width_ratio")
    distance_deadzone = float(control.get("track_target_distance_deadzone_ratio", 0.08))

    if isinstance(box_width_ratio, (int, float)) and desired_box_width_ratio > 0.0:
        distance_error = desired_box_width_ratio - float(box_width_ratio)
        normalized_error = _clamp(distance_error / desired_box_width_ratio, -1.0, 1.0)
        if abs(normalized_error) > distance_deadzone:
            command["vx"] = normalized_error * float(control["linear_speed_mps"])
    else:
        distance = state.get("distance", "unknown")
        if distance == "far":
            command["vx"] = float(control["linear_speed_mps"])
        elif distance == "near":
            command["vx"] = 0.0

    command["vx"] += forward_bias * float(control["linear_speed_mps"]) * 0.5
    return command


def map_intent_to_command(intent_output: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    config = _load_parameters()
    control = config["control"]
    drone = config["drone"]

    intent = str(intent_output.get("intent", "hold")).strip().lower() or "hold"
    intent_params = intent_output.get("params", {})
    if not isinstance(intent_params, dict):
        intent_params = {}

    if state.get("obstacle") and intent != "land":
        return _safe_hover_command(float(control["command_duration_s"]))

    if intent == "takeoff":
        return {
            "command": "takeoff",
            "altitude_m": float(drone["takeoff_altitude_m"]),
        }

    if intent == "land":
        return {"command": "land"}

    if intent == "hold":
        return _safe_hover_command(float(control["command_duration_s"]))

    if intent == "move_relative":
        return _apply_safety_limits(_move_relative_command(intent_params, config), config)

    if intent == "track_target":
        return _apply_safety_limits(_track_target_command(intent_params, state, config), config)

    return _safe_hover_command(float(control["command_duration_s"]))


def map_action_to_command(action: str, state: Dict[str, Any]) -> Dict[str, Any]:
    legacy_to_intent = {
        "takeoff": {"intent": "takeoff", "params": {}},
        "land": {"intent": "land", "params": {}},
        "hover": {"intent": "hold", "params": {}},
        "move_forward": {"intent": "move_relative", "params": {"forward": 1.0}},
        "move_backward": {"intent": "move_relative", "params": {"forward": -1.0}},
        "turn_left": {"intent": "move_relative", "params": {"yaw": -1.0}},
        "turn_right": {"intent": "move_relative", "params": {"yaw": 1.0}},
        "follow_person": {"intent": "track_target", "params": {"target": "person", "continuous": True}},
    }

    intent_output = legacy_to_intent.get(action, {"intent": "hold", "params": {}})
    return map_intent_to_command(intent_output, state)
