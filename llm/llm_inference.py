from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
from urllib import error, request


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "parameters.json"
PROMPT_PATH = Path(__file__).resolve().with_name("prompt.txt")
ALLOWED_ACTIONS = {
    "takeoff",
    "land",
    "hover",
    "move_forward",
    "move_backward",
    "turn_left",
    "turn_right",
    "follow_person",
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


def _normalize_llm_output(raw_output: Dict[str, Any]) -> Dict[str, Any]:
    action = str(raw_output.get("action", "")).strip().lower()
    if action not in ALLOWED_ACTIONS:
        raise ValueError(f"Unsupported action from LLM: {action}")

    reason = str(raw_output.get("reason", "")).strip() or "No reason provided."
    return {
        "action": action,
        "reason": reason,
        "decision_source": "ollama",
    }


def _rule_based_decision(state: Dict[str, Any]) -> Dict[str, str]:
    command = _normalize_command(state.get("user_command", ""))
    obstacle = state.get("obstacle", False)
    person_detected = state.get("person_detected", False)

    if _contains_any(command, ("land", "착륙", "내려")):
        return {"action": "land", "reason": "User requested landing.", "decision_source": "rule_fallback"}
    if _contains_any(command, ("stop", "hover", "wait", "hold", "멈춰", "정지", "스탑")):
        return {"action": "hover", "reason": "User requested a stop or hold.", "decision_source": "rule_fallback"}
    if _contains_any(command, ("take off", "takeoff", "이륙")):
        return {"action": "takeoff", "reason": "User requested takeoff.", "decision_source": "rule_fallback"}
    if obstacle:
        return {
            "action": "hover",
            "reason": "Obstacle detected, holding position for safety.",
            "decision_source": "rule_fallback",
        }
    if _contains_any(command, ("follow", "track", "따라")):
        if person_detected:
            return {
                "action": "follow_person",
                "reason": "User requested following a detected person.",
                "decision_source": "rule_fallback",
            }
        return {"action": "hover", "reason": "No person detected to follow.", "decision_source": "rule_fallback"}
    if _contains_any(command, ("left", "왼쪽", "좌측", "좌회전")):
        return {"action": "turn_left", "reason": "User requested movement to the left.", "decision_source": "rule_fallback"}
    if _contains_any(command, ("right", "오른쪽", "우측", "우회전")):
        return {"action": "turn_right", "reason": "User requested movement to the right.", "decision_source": "rule_fallback"}
    if _contains_any(command, ("forward", "ahead", "front", "앞", "전진")):
        return {"action": "move_forward", "reason": "User requested forward movement.", "decision_source": "rule_fallback"}
    if _contains_any(command, ("backward", "back", "뒤", "후진")):
        return {"action": "move_backward", "reason": "User requested backward movement.", "decision_source": "rule_fallback"}
    return {
        "action": "hover",
        "reason": "Command was unclear, defaulting to safe hover.",
        "decision_source": "rule_fallback",
    }


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
    return _normalize_llm_output(_extract_json(response_text))


def decide_action(state: Dict[str, Any]) -> Dict[str, Any]:
    config = _load_config()["llm"]

    try:
        return _ollama_decision(state)
    except (error.URLError, TimeoutError, ValueError, json.JSONDecodeError, OSError) as exc:
        if config.get("use_rule_fallback", True):
            fallback = _rule_based_decision(state)
            fallback["fallback_reason"] = f"Ollama unavailable or invalid response: {exc}"
            return fallback
        raise


if __name__ == "__main__":
    demo_state = {
        "user_command": "Follow me",
        "gesture": "none",
        "person_detected": True,
        "person_position": "left",
        "distance": "far",
        "obstacle": False,
    }
    print(json.dumps(decide_action(demo_state), ensure_ascii=False, indent=2))
