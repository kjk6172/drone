from __future__ import annotations

import json
from pathlib import Path

from pymavlink import mavutil


CONFIG_PATH = Path(__file__).resolve().parent / "config" / "parameters.json"


def load_pixhawk_config() -> tuple[str, int]:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = json.load(file)
    drone = config["drone"]
    return drone["port"], drone["baud"]


def main() -> None:
    try:
        import serial  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "pyserial is not installed. Run 'pip uninstall serial -y' and 'pip install pyserial'."
        ) from exc

    if not hasattr(serial, "SerialException"):
        raise RuntimeError(
            "Wrong 'serial' package detected. Uninstall 'serial' and install 'pyserial' instead."
        )

    port, baud = load_pixhawk_config()
    print(f"Connecting to Pixhawk on {port} at {baud} baud...")
    master = mavutil.mavlink_connection(port, baud=baud)

    print("Waiting for heartbeat...")
    master.wait_heartbeat(timeout=10)
    print("Connected to Pixhawk!")
    print(f"Target system: {master.target_system}")
    print(f"Target component: {master.target_component}")

    while True:
        msg = master.recv_match(type="HEARTBEAT", blocking=True, timeout=5)
        if msg:
            print("Heartbeat received")
            print(f"Base mode: {msg.base_mode}")
            print(f"Custom mode: {msg.custom_mode}")


if __name__ == "__main__":
    main()
