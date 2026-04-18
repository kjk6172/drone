from __future__ import annotations

import argparse
import json
import math
import socket
import time
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from pymavlink import mavutil
except ImportError:  # pragma: no cover - optional runtime dependency
    mavutil = None


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "parameters.json"
_PIXHAWK_CONTROLLER = None


def _load_config() -> Dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


class PixhawkController:
    def __init__(self, config: Dict[str, Any]) -> None:
        self._config = config
        self._master = None

    @property
    def _drone(self) -> Dict[str, Any]:
        return self._config["drone"]

    @property
    def _safety(self) -> Dict[str, Any]:
        return self._config.get("safety", {})

    def connect(self) -> None:
        if self._master is not None:
            return

        if mavutil is None:
            raise RuntimeError("pymavlink is not installed.")

        connection_string = str(self._drone.get("connection_string") or self._drone.get("port") or "").strip()
        if not connection_string:
            raise RuntimeError("Drone connection string/port is empty.")

        baud = int(self._drone.get("baud", 115200))
        heartbeat_timeout = float(self._drone.get("heartbeat_timeout_s", 15.0))

        try:
            self._master = mavutil.mavlink_connection(connection_string, baud=baud)
            print(f"[Pixhawk] Connecting to {connection_string} (baud={baud})...")
            self._master.wait_heartbeat(timeout=heartbeat_timeout)
        except Exception as exc:
            self._master = None
            raise RuntimeError(
                "Failed MAVLink connection. "
                "Check port/baud and Python serial stack (`pip install pyserial`, remove conflicting `serial` package). "
                f"Details: {exc}"
            ) from exc
        print(
            f"[Pixhawk] Heartbeat received: system={self._master.target_system}, component={self._master.target_component}"
        )

    def _command_long(
        self,
        command: int,
        param1: float = 0.0,
        param2: float = 0.0,
        param3: float = 0.0,
        param4: float = 0.0,
        param5: float = 0.0,
        param6: float = 0.0,
        param7: float = 0.0,
    ) -> None:
        self.connect()
        assert self._master is not None
        self._master.mav.command_long_send(
            self._master.target_system,
            self._master.target_component,
            command,
            0,
            float(param1),
            float(param2),
            float(param3),
            float(param4),
            float(param5),
            float(param6),
            float(param7),
        )

    def _set_mode(self, mode_name: str) -> bool:
        self.connect()
        assert self._master is not None

        mapping = self._master.mode_mapping()
        if not mapping:
            print(f"[Pixhawk] Mode mapping unavailable. Cannot set mode '{mode_name}'.")
            return False

        mode_id = mapping.get(mode_name)
        if mode_id is None:
            print(f"[Pixhawk] Mode '{mode_name}' not found. Available: {sorted(mapping.keys())}")
            return False

        self._master.set_mode(mode_id)
        time.sleep(0.1)
        return True

    def arm(self) -> None:
        self.connect()
        self._command_long(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        time.sleep(float(self._drone.get("arm_settle_seconds", 1.0)))
        print("[Pixhawk] Arm command sent.")

    def takeoff(self, altitude_m: float) -> None:
        guided_mode = str(self._drone.get("guided_mode", "GUIDED"))
        self._set_mode(guided_mode)

        if bool(self._drone.get("arm_before_takeoff", True)):
            self.arm()

        self._command_long(
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            param7=float(altitude_m),
        )
        print(f"[Pixhawk] Takeoff command sent (altitude={altitude_m:.2f}m).")

    def land(self) -> None:
        land_mode = str(self._drone.get("land_mode", "LAND"))
        mode_ok = self._set_mode(land_mode)
        if not mode_ok:
            self._command_long(mavutil.mavlink.MAV_CMD_NAV_LAND)
        print("[Pixhawk] Land command sent.")

    def send_body_velocity(self, vx: float, vy: float, vz_up: float, yaw_rate_degps: float, duration_s: float) -> None:
        self.connect()
        assert self._master is not None

        guided_mode = str(self._drone.get("guided_mode", "GUIDED"))
        self._set_mode(guided_mode)

        max_linear = float(self._safety.get("max_linear_speed_mps", 1.0))
        max_vertical = float(self._safety.get("max_vertical_speed_mps", 0.5))
        max_yaw = float(self._safety.get("max_yaw_rate_degps", 45.0))
        min_duration = float(self._safety.get("min_command_duration_s", 0.1))
        max_duration = float(self._safety.get("max_command_duration_s", 2.0))

        vx = _clamp(float(vx), -max_linear, max_linear)
        vy = _clamp(float(vy), -max_linear, max_linear)
        vz_up = _clamp(float(vz_up), -max_vertical, max_vertical)
        yaw_rate_degps = _clamp(float(yaw_rate_degps), -max_yaw, max_yaw)
        duration_s = _clamp(float(duration_s), min_duration, max_duration)

        # MAV_FRAME_BODY_NED uses +Z down. Internal command uses +Z up.
        vz_ned = -vz_up
        yaw_rate_radps = math.radians(yaw_rate_degps)

        type_mask = (
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
        )

        send_hz = float(self._drone.get("setpoint_rate_hz", 5.0))
        send_period = max(0.05, 1.0 / max(send_hz, 1.0))
        iterations = max(1, int(math.ceil(duration_s / send_period)))

        for _ in range(iterations):
            self._master.mav.set_position_target_local_ned_send(
                0,
                self._master.target_system,
                self._master.target_component,
                mavutil.mavlink.MAV_FRAME_BODY_NED,
                type_mask,
                0.0,
                0.0,
                0.0,
                vx,
                vy,
                vz_ned,
                0.0,
                0.0,
                0.0,
                0.0,
                yaw_rate_radps,
            )
            time.sleep(send_period)

        print(
            "[Pixhawk] Velocity command sent "
            f"vx={vx:.2f}, vy={vy:.2f}, vz_up={vz_up:.2f}, yaw_rate={yaw_rate_degps:.2f}, duration={duration_s:.2f}s"
        )


def _get_pixhawk_controller() -> PixhawkController:
    global _PIXHAWK_CONTROLLER

    if _PIXHAWK_CONTROLLER is None:
        _PIXHAWK_CONTROLLER = PixhawkController(_load_config())

    return _PIXHAWK_CONTROLLER


def execute_command(command: Dict[str, Any]) -> None:
    if mavutil is None:
        print(f"[Dry Run] Received command (pymavlink unavailable): {command}")
        return

    controller = _get_pixhawk_controller()
    try:
        command_type = str(command.get("command", "")).strip().lower()
        if command_type == "takeoff":
            altitude_m = float(command.get("altitude_m", _load_config()["drone"].get("takeoff_altitude_m", 2.0)))
            controller.takeoff(altitude_m)
            return

        if command_type == "land":
            controller.land()
            return

        if command_type == "hover":
            duration = float(command.get("duration", _load_config()["control"].get("command_duration_s", 1.0)))
            controller.send_body_velocity(0.0, 0.0, 0.0, 0.0, duration)
            return

        if command_type == "velocity":
            controller.send_body_velocity(
                vx=float(command.get("vx", 0.0)),
                vy=float(command.get("vy", 0.0)),
                vz_up=float(command.get("vz", 0.0)),
                yaw_rate_degps=float(command.get("yaw_rate", 0.0)),
                duration_s=float(command.get("duration", _load_config()["control"].get("command_duration_s", 1.0))),
            )
            return

        print(f"[Warning] Unsupported command type: {command_type} (payload={command})")
    except Exception as exc:
        print(f"[Error] Pixhawk execution failed. Falling back to dry-run log. reason={exc}")
        print(f"[Dry Run] Received command: {command}")


def start_server(host: Optional[str] = None, port: Optional[int] = None) -> None:
    config = _load_config()["communication"]
    server_host = host or config.get("listen_host") or config["host"]
    server_port = int(port or config["port"])

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
                    chunk = connection.recv(int(config["buffer_size"]))
                    if not chunk:
                        break
                    buffer += chunk.decode("utf-8")
                    while "\n" in buffer:
                        raw_message, buffer = buffer.split("\n", 1)
                        if not raw_message.strip():
                            continue
                        try:
                            payload = json.loads(raw_message)
                            execute_command(payload)
                        except json.JSONDecodeError as exc:
                            print(f"[Warning] Invalid JSON payload: {exc}; raw={raw_message!r}")
                        except Exception as exc:
                            print(f"[Error] Failed to execute command: {exc}")


def _build_argument_parser() -> argparse.ArgumentParser:
    config = _load_config()["communication"]
    parser = argparse.ArgumentParser(description="Receive and execute drone commands on Pixhawk.")
    parser.add_argument("--host", default=config.get("listen_host") or config["host"], help="Bind host")
    parser.add_argument("--port", type=int, default=config["port"], help="Bind port")
    return parser


if __name__ == "__main__":
    args = _build_argument_parser().parse_args()
    start_server(host=args.host, port=args.port)
