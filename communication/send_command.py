from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any, Dict


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "parameters.json"


def _load_communication_config() -> Dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = json.load(file)
    return config["communication"]


def send_command(command: Dict[str, Any], host: str | None = None, port: int | None = None) -> None:
    config = _load_communication_config()
    target_host = host or config["host"]
    target_port = port or config["port"]
    message = (json.dumps(command) + "\n").encode("utf-8")

    with socket.create_connection((target_host, target_port), timeout=5) as sock:
        sock.sendall(message)


if __name__ == "__main__":
    demo_command = {"command": "hover", "duration": 1.0}
    send_command(demo_command)
    print(f"Sent command: {demo_command}")
