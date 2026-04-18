from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
from urllib import error, request


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "parameters.json"
PROMPT_PATH = Path(__file__).resolve().with_name("prompt.txt")
ALLOWED_INTENTS = {
    "takeoff",
    "land",
    "hold",
    "move_relative",
    "track_target",
}
LEGACY_ACTION_TO_INTENT = {
    "takeoff": "takeoff",
    "land": "land",
    "hover": "hold",
    "move_forward": "move_relative",
    "move_backward": "move_relative",
    "turn_left": "move_relative",
    "turn_right": "move_relative",
    "follow_person": "track_target",
}


def _load_config() -> Dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def _load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8").strip()


def _extract_json(content: str) -> Dict[str, Any]:
    content = content.strip()
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in LLM response.")
    return json.loads(content[start : end + 1])


def _normalize_command(command: str) -> str:
    return command.lower().strip().replace("_", " ")


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


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


def _read_bool(payload: Dict[str, Any], key: str, default: bool) -> bool:
    value = payload.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    return default


def _normalize_move_relative_params(raw_params: Dict[str, Any], default_duration_s: float) -> Dict[str, Any]:
    alias = {
        "forward": "forward",
        "backward": "forward",
        "right": "right",
        "left": "right",
        "up": "up",
        "down": "up",
        "yaw": "yaw",
    }

    normalized = {
        "forward": 0.0,
        "right": 0.0,
        "up": 0.0,
        "yaw": 0.0,
        "duration_s": default_duration_s,
    }

    for key, canonical in alias.items():
        if key not in raw_params:
            continue

        value = _read_float(raw_params, key, 0.0)
        if key in {"backward", "left", "down"}:
            value = -value
        normalized[canonical] = _clamp(value, -1.0, 1.0)

    normalized["duration_s"] = max(0.1, _read_float(raw_params, "duration_s", default_duration_s))
    return normalized


def _normalize_track_target_params(raw_params: Dict[str, Any], default_box_ratio: float) -> Dict[str, Any]:
    target = str(raw_params.get("target", "person")).strip().lower() or "person"
    if target != "person":
        target = "person"

    desired_box_width_ratio = _clamp(
        _read_float(raw_params, "desired_box_width_ratio", default_box_ratio),
        0.1,
        0.8,
    )

    forward_bias = _clamp(_read_float(raw_params, "forward_bias", 0.0), -1.0, 1.0)

    return {
        "target": target,
        "continuous": _read_bool(raw_params, "continuous", True),
        "desired_box_width_ratio": round(desired_box_width_ratio, 3),
        "forward_bias": round(forward_bias, 3),
    }


def _normalize_llm_output(raw_output: Dict[str, Any], decision_source: str) -> Dict[str, Any]:
    config = _load_config()
    control = config["control"]

    raw_intent = str(raw_output.get("intent", "")).strip().lower()
    if not raw_intent:
        raw_action = str(raw_output.get("action", "")).strip().lower()
        raw_intent = LEGACY_ACTION_TO_INTENT.get(raw_action, "")

    if raw_intent not in ALLOWED_INTENTS:
        raise ValueError(f"Unsupported intent from LLM: {raw_intent}")

    raw_params = raw_output.get("params", {})
    if not isinstance(raw_params, dict):
        raw_params = {}

    if raw_intent == "move_relative":
        params = _normalize_move_relative_params(raw_params, control["command_duration_s"])
    elif raw_intent == "track_target":
        params = _normalize_track_target_params(raw_params, control["track_target_box_width_ratio"])
    else:
        params = {}

    reason = str(raw_output.get("reason", "")).strip() or "No reason provided."

    return {
        "intent": raw_intent,
        "params": params,
        "reason": reason,
        "decision_source": decision_source,
    }


def _rule_based_decision(state: Dict[str, Any]) -> Dict[str, Any]:
    command = _normalize_command(state.get("user_command", ""))
    obstacle = bool(state.get("obstacle", False))

    if _contains_any(command, ("land", "touch down", "descend and land")):
        return {
            "intent": "land",
            "params": {},
            "reason": "User requested landing.",
            "decision_source": "rule_fallback",
        }

    if _contains_any(command, ("take off", "takeoff", "lift off", "launch")):
        return {
            "intent": "takeoff",
            "params": {},
            "reason": "User requested takeoff.",
            "decision_source": "rule_fallback",
        }

    if _contains_any(command, ("stop", "hover", "wait", "hold", "freeze")):
        return {
            "intent": "hold",
            "params": {},
            "reason": "User requested hold position.",
            "decision_source": "rule_fallback",
        }

    if obstacle:
        return {
            "intent": "hold",
            "params": {},
            "reason": "Obstacle detected, holding position for safety.",
            "decision_source": "rule_fallback",
        }

    if _contains_any(command, ("follow", "track", "chase")):
        return {
            "intent": "track_target",
            "params": {
                "target": "person",
                "continuous": True,
            },
            "reason": "User requested tracking a person.",
            "decision_source": "rule_fallback",
        }

    speed_scale = 1.0
    if _contains_any(command, ("slight", "slightly", "a bit", "little")):
        speed_scale = 0.4
    elif _contains_any(command, ("slow", "slowly", "gentle")):
        speed_scale = 0.6

    direction = {
        "forward": 0.0,
        "right": 0.0,
        "up": 0.0,
        "yaw": 0.0,
    }

    if _contains_any(command, ("forward", "ahead", "in front")):
        direction["forward"] += 1.0
    if _contains_any(command, ("backward", "back", "reverse")):
        direction["forward"] -= 1.0
    if _contains_any(command, ("right", "rightward", "strafe right")):
        direction["right"] += 1.0
    if _contains_any(command, ("left", "leftward", "strafe left")):
        direction["right"] -= 1.0
    if _contains_any(command, ("up", "ascend", "rise", "higher")):
        direction["up"] += 1.0
    if _contains_any(command, ("down", "descend", "lower")):
        direction["up"] -= 1.0
    if _contains_any(command, ("turn right", "yaw right", "rotate right", "clockwise")):
        direction["yaw"] += 1.0
    if _contains_any(command, ("turn left", "yaw left", "rotate left", "counterclockwise")):
        direction["yaw"] -= 1.0

    if any(abs(axis_value) > 0.0 for axis_value in direction.values()):
        params = {
            "forward": round(_clamp(direction["forward"] * speed_scale, -1.0, 1.0), 3),
            "right": round(_clamp(direction["right"] * speed_scale, -1.0, 1.0), 3),
            "up": round(_clamp(direction["up"] * speed_scale, -1.0, 1.0), 3),
            "yaw": round(_clamp(direction["yaw"] * speed_scale, -1.0, 1.0), 3),
            "duration_s": 1.0,
        }
        return {
            "intent": "move_relative",
            "params": params,
            "reason": "User requested directional movement.",
            "decision_source": "rule_fallback",
        }

    return {
        "intent": "hold",
        "params": {},
        "reason": "Command was unclear, defaulting to safe hold.",
        "decision_source": "rule_fallback",
    }


def _apply_directional_guardrails(state: Dict[str, Any], intent_output: Dict[str, Any]) -> Dict[str, Any]:
    if intent_output.get("intent") != "move_relative":
        return intent_output

    params = intent_output.get("params", {})
    if not isinstance(params, dict):
        return intent_output

    command = _normalize_command(state.get("user_command", ""))

    def has_any(words: tuple[str, ...]) -> bool:
        return _contains_any(command, words)

    axis_rules = (
        ("forward", ("forward", "ahead", "in front"), ("backward", "back", "reverse")),
        ("right", ("right", "rightward", "strafe right"), ("left", "leftward", "strafe left")),
        ("up", ("up", "ascend", "rise", "higher"), ("down", "descend", "lower")),
        ("yaw", ("turn right", "yaw right", "rotate right", "clockwise"), ("turn left", "yaw left", "rotate left", "counterclockwise")),
    )

    guarded_params = dict(params)
    for axis, positive_words, negative_words in axis_rules:
        axis_value = _read_float(guarded_params, axis, 0.0)
        has_positive = has_any(positive_words)
        has_negative = has_any(negative_words)

        if has_positive and not has_negative and axis_value < 0:
            guarded_params[axis] = abs(axis_value)
        elif has_negative and not has_positive and axis_value > 0:
            guarded_params[axis] = -abs(axis_value)
        elif not has_positive and not has_negative:
            guarded_params[axis] = 0.0

    output = dict(intent_output)
    output["params"] = guarded_params
    return output


def _ollama_decision(state: Dict[str, Any]) -> Dict[str, Any]:
    config = _load_config()["llm"]
    prompt = _load_prompt()

    payload = {
        "model": config["model"],
        "prompt": (
            f"{prompt}\n\n"
            f"Current state JSON:\n{json.dumps(state, ensure_ascii=False, indent=2)}\n\n"
            "Return the decision JSON now."
        ),
        "stream": False,
        "format": "json",
    }

    request_data = json.dumps(payload).encode("utf-8")
    http_request = request.Request(
        config["endpoint"],
        data=request_data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with request.urlopen(http_request, timeout=config["timeout_seconds"]) as response:
        body = json.loads(response.read().decode("utf-8"))

    response_text = body.get("response", "{}")
    return _normalize_llm_output(_extract_json(response_text), decision_source="ollama")


def decide_intent(state: Dict[str, Any]) -> Dict[str, Any]:
    config = _load_config()["llm"]

    try:
        ollama_output = _ollama_decision(state)
        return _apply_directional_guardrails(state, ollama_output)
    except (error.URLError, TimeoutError, ValueError, json.JSONDecodeError, OSError) as exc:
        if config.get("use_rule_fallback", True):
            fallback = _rule_based_decision(state)
            fallback["fallback_reason"] = f"Ollama unavailable or invalid response: {exc}"
            return _apply_directional_guardrails(state, fallback)
        raise


def _intent_to_legacy_action(intent_output: Dict[str, Any]) -> str:
    intent = intent_output["intent"]
    params = intent_output.get("params", {})

    if intent == "takeoff":
        return "takeoff"
    if intent == "land":
        return "land"
    if intent == "hold":
        return "hover"
    if intent == "track_target":
        return "follow_person"
    if intent != "move_relative":
        return "hover"

    forward = float(params.get("forward", 0.0))
    right = float(params.get("right", 0.0))
    yaw = float(params.get("yaw", 0.0))

    dominant_axis = max(
        (("forward", abs(forward)), ("right", abs(right)), ("yaw", abs(yaw))),
        key=lambda item: item[1],
    )[0]

    if dominant_axis == "forward":
        return "move_forward" if forward >= 0 else "move_backward"
    if dominant_axis == "right":
        return "turn_right" if right >= 0 else "turn_left"
    return "turn_right" if yaw >= 0 else "turn_left"


def decide_action(state: Dict[str, Any]) -> Dict[str, Any]:
    intent_output = decide_intent(state)
    return {
        "action": _intent_to_legacy_action(intent_output),
        "reason": intent_output.get("reason", ""),
        "decision_source": intent_output.get("decision_source", "unknown"),
        "params": intent_output.get("params", {}),
    }


if __name__ == "__main__":
    demo_state = {
        "user_command": "Follow me",
        "gesture": "none",
        "person_detected": True,
        "person_position": "left",
        "distance": "far",
        "obstacle": False,
    }
    print(json.dumps(decide_intent(demo_state), ensure_ascii=False, indent=2))
