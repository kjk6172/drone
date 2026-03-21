from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any, Dict


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "parameters.json"


def _load_config() -> Dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def execute_command(command: Dict[str, Any]) -> None:
    mavsdk_available = False
    try:
        from mavsdk import System  # type: ignore # noqa: F401

        mavsdk_available = True
    except ImportError:
        mavsdk_available = False

    if mavsdk_available:
        print(f"[MAVSDK placeholder] Execute command: {command}")
        return

    print(f"[Dry Run] Received command: {command}")


def start_server(host: str | None = None, port: int | None = None) -> None:
    config = _load_config()["communication"]
    server_host = host or config["host"]
    server_port = port or config["port"]

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((server_host, server_port))
        server.listen(1)
        print(f"Listening for commands on {server_host}:{server_port}")

        while True:
            connection, address = server.accept()
            with connection:
                print(f"Connected by {address}")
                buffer = ""
                while True:
                    chunk = connection.recv(config["buffer_size"])
                    if not chunk:
                        break
                    buffer += chunk.decode("utf-8")
                    while "\n" in buffer:
                        raw_message, buffer = buffer.split("\n", 1)
                        if raw_message.strip():
                            execute_command(json.loads(raw_message))


if __name__ == "__main__":
    start_server()
