"""
Layer 3 -- hardware bridge (opt-in, isolated from the sim).

Drives the physical 3-DOF arm (Feetech SCS0009 servos, IDs 1/2/3) from the
SAME MotorVector contract the sim consumes (arm.motor_contract). This module
must never be able to crash Layer 2: import failures, missing SDK, or a
disconnected board all degrade to a clear, catchable ArmHardwareError instead
of raising SystemExit or an unguarded exception. `arm/sim.py` never imports
this module, and nothing in here imports mujoco or the RL agents.

Hardware settings (confirmed working on this rig):
    protocol_end = 1
    baudrate     = 1_000_000
    reg 40       = torque enable
    reg 41       = goal acceleration
    reg 42       = goal position (0-1023)
    reg 46       = goal speed
    servo IDs: 1 = base, 2 = shoulder, 3 = claw

`find_port()` and `scservo_sdk` live in a separate personal repo
(Feetech-Servo-SDK), not inside this project. Point FEETECH_SDK_ROOT /
FIND_PORT_DIR at wherever that repo lives on this machine; env vars
override the defaults below if it ever moves.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

from arm.motor_contract import MotorVector, to_scs_positions

FEETECH_SDK_ROOT = os.environ.get(
    "FEETECH_SDK_ROOT", "/Users/liz/Documents/Feetech-Servo-SDK"
)
FIND_PORT_DIR = os.environ.get(
    "FIND_PORT_DIR", os.path.join(FEETECH_SDK_ROOT, "scsservo_sdk_example")
)

for _p in (FEETECH_SDK_ROOT, FIND_PORT_DIR):
    if _p and os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)


class ArmHardwareError(Exception):
    """Raised for any hardware-bridge failure: missing SDK, no port, bad write."""


try:
    from scservo_sdk import PacketHandler, PortHandler  # type: ignore
    from find_port import find_port  # type: ignore

    HARDWARE_SDK_AVAILABLE = True
    _import_error: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - depends on local machine setup
    HARDWARE_SDK_AVAILABLE = False
    _import_error = exc

BAUDRATE = 1_000_000
PROTOCOL_END = 1

ADDR_TORQUE_ENABLE = 40
ADDR_GOAL_ACC = 41
ADDR_GOAL_POSITION = 42
ADDR_GOAL_SPEED = 46

# Conservative defaults so a bad model output can't whip the arm around.
DEFAULT_MOVING_SPEED = 300
DEFAULT_MOVING_ACC = 30

SERVO_IDS = (1, 2, 3)


class ArmHardware:
    """Context manager for the real 3-DOF arm. Opt-in -- the sim never touches this."""

    def __init__(
        self,
        port: Optional[str] = None,
        moving_speed: int = DEFAULT_MOVING_SPEED,
        moving_acc: int = DEFAULT_MOVING_ACC,
    ):
        if not HARDWARE_SDK_AVAILABLE:
            raise ArmHardwareError(
                "scservo_sdk / find_port are not importable "
                f"(FEETECH_SDK_ROOT={FEETECH_SDK_ROOT!r}, "
                f"FIND_PORT_DIR={FIND_PORT_DIR!r}). "
                f"Original import error: {_import_error!r}"
            )
        self._port_name = port
        self._moving_speed = int(moving_speed)
        self._moving_acc = int(moving_acc)
        self._port_handler = None
        self._packet_handler = None

    def connect(self) -> "ArmHardware":
        try:
            self._port_name = self._port_name or find_port()
        except SystemExit as exc:
            raise ArmHardwareError(
                "No serial port found -- is the URT board plugged in and the "
                "CH340 driver loaded?"
            ) from exc

        self._port_handler = PortHandler(self._port_name)
        self._packet_handler = PacketHandler(PROTOCOL_END)

        if not self._port_handler.openPort():
            raise ArmHardwareError(f"Failed to open port {self._port_name!r}")
        if not self._port_handler.setBaudRate(BAUDRATE):
            self._port_handler.closePort()
            raise ArmHardwareError(
                f"Failed to set baudrate {BAUDRATE} on {self._port_name!r}"
            )

        for servo_id in SERVO_IDS:
            self._packet_handler.write1ByteTxRx(
                self._port_handler, servo_id, ADDR_TORQUE_ENABLE, 1
            )
            self._packet_handler.write1ByteTxRx(
                self._port_handler, servo_id, ADDR_GOAL_ACC, self._moving_acc
            )
            self._packet_handler.write2ByteTxRx(
                self._port_handler, servo_id, ADDR_GOAL_SPEED, self._moving_speed
            )

        print(f"[hardware] connected on {self._port_name}, torque enabled on {SERVO_IDS}")
        return self

    def send_motor_values(self, motors: MotorVector) -> None:
        """Write one normalized motor vector to the servos, safe-range clamped."""
        if self._port_handler is None:
            raise ArmHardwareError("send_motor_values() called before connect()")
        for servo_id, position in to_scs_positions(motors).items():
            self._packet_handler.write2ByteTxRx(
                self._port_handler, servo_id, ADDR_GOAL_POSITION, position
            )

    def close(self) -> None:
        if self._port_handler is not None:
            self._port_handler.closePort()
            self._port_handler = None
            print("[hardware] port closed")

    def __enter__(self) -> "ArmHardware":
        return self.connect()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def try_connect(port: Optional[str] = None) -> Optional["ArmHardware"]:
    """Best-effort connect that never raises: returns None with a clear message on failure."""
    try:
        return ArmHardware(port).connect()
    except ArmHardwareError as exc:
        print(f"[hardware] unavailable: {exc}")
        return None
