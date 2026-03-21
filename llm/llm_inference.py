from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
from urllib import error, request


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "parameters.json"
PROMPT_PATH = Path(__file__).resolve().with_name("prompt.txt")


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


def _rule_based_decision(state: Dict[str, Any]) -> Dict[str, str]:
    command = state.get("user_command", "").lower()
    obstacle = state.get("obstacle", False)
    person_detected = state.get("person_detected", False)

    if "land" in command:
        return {"action": "land", "reason": "User requested landing."}
    if any(keyword in command for keyword in ("stop", "hover", "wait", "hold")):
        return {"action": "hover", "reason": "User requested a stop or hold."}
    if any(keyword in command for keyword in ("take off", "takeoff")):
        return {"action": "takeoff", "reason": "User requested takeoff."}
    if obstacle:
        return {"action": "hover", "reason": "Obstacle detected, holding position for safety."}
    if "follow" in command:
        if person_detected:
            return {"action": "follow_person", "reason": "User requested following a detected person."}
        return {"action": "hover", "reason": "No person detected to follow."}
    if "left" in command:
        return {"action": "turn_left", "reason": "User requested movement to the left."}
    if "right" in command:
        return {"action": "turn_right", "reason": "User requested movement to the right."}
    if any(keyword in command for keyword in ("forward", "ahead", "front")):
        return {"action": "move_forward", "reason": "User requested forward movement."}
    if any(keyword in command for keyword in ("backward", "back")):
        return {"action": "move_backward", "reason": "User requested backward movement."}
    return {"action": "hover", "reason": "Command was unclear, defaulting to safe hover."}


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
    return _extract_json(response_text)


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
