"""
Motor-value contract
=====================
The single stable interface between the RL model, the MuJoCo simulator, and
the physical servo hardware:

    MotorVector = [base, shoulder, claw], each normalized to [0.0, 1.0]

Every other layer (model_bridge, sim, hardware) consumes or produces this
type and nothing else. Change the model, the sim, or the hardware
independently as long as they still speak this contract.

This module also owns the safe per-joint sub-ranges used to clamp values
before they ever reach a physical servo, and the conversion functions into
MuJoCo joint angles and SCS0009 servo positions.
"""

from __future__ import annotations

from dataclasses import dataclass

JOINT_NAMES = ("base", "shoulder", "claw")

# ---------------------------------------------------------------------------
# Safe ranges -- EDIT THESE to match your physical arm's real range of motion.
# ---------------------------------------------------------------------------

# Normalized [0, 1] sub-range each joint's motor value is clamped to before
# any conversion happens. Keeps a bad/out-of-distribution model output from
# ever requesting the extreme ends of a joint's travel.
SAFE_RANGE_NORM: dict[str, tuple[float, float]] = {
    "base": (0.05, 0.95),
    "shoulder": (0.10, 0.90),
    "claw": (0.0, 1.0),
}

# MuJoCo joint value range that normalized [0, 1] maps onto. Units depend on
# the joint type in arm/mujoco_scene.py: radians for the base/shoulder hinges,
# meters for the claw's slide (finger) joints.
MUJOCO_ANGLE_RANGE: dict[str, tuple[float, float]] = {
    "base": (-1.57, 1.57),     # +/- 90 degrees of base rotation
    # The object always sits on the table (ground level), never in the air,
    # so the WHOLE shoulder range stays tilted down toward the table --
    # from a deep reach (-0.39 rad) to a shallower one (-0.15 rad) -- rather
    # than sweeping up above horizontal. Positive would lift the arm UP (see
    # arm/mujoco_scene.py for the axis convention); we never use positive
    # here on purpose. -0.39 rad leaves a few cm of fingertip clearance above
    # the floor -- comfortably more than the sim's cosmetic tremor (see
    # arm/sim.py's TREMOR_AMPLITUDE) ever dips into -- so the arm can reach
    # all the way down to the object without ever clipping through the table.
    "shoulder": (-0.39, -0.15),
    "claw": (0.0, 0.035),       # gripper finger travel in meters, 0 = open
}

# SCS0009 servo goal-position range (raw units, 0-1023) that normalized
# [0, 1] maps onto, per joint. These are the values written to register 42.
# Keep well inside 0-1023 and inside your arm's mechanically safe travel.
SCS_POSITION_RANGE: dict[str, tuple[int, int]] = {
    "base": (200, 824),
    "shoulder": (256, 768),
    "claw": (300, 700),
}

SCS_MIN, SCS_MAX = 0, 1023

# Servo IDs on the daisy chain, per joint.
SERVO_ID: dict[str, int] = {
    "base": 1,
    "shoulder": 2,
    "claw": 3,
}


@dataclass(frozen=True)
class MotorVector:
    """A normalized [base, shoulder, claw] motor command, each in [0.0, 1.0]."""

    base: float
    shoulder: float
    claw: float

    def __post_init__(self) -> None:
        for name, value in zip(JOINT_NAMES, self.as_tuple()):
            if not (0.0 <= float(value) <= 1.0):
                raise ValueError(
                    f"MotorVector.{name}={value!r} is out of the required [0.0, 1.0] range"
                )

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.base, self.shoulder, self.claw)

    def as_dict(self) -> dict[str, float]:
        return dict(zip(JOINT_NAMES, self.as_tuple()))

    def clamped(self) -> "MotorVector":
        """Return a copy clamped to the editable safe sub-range per joint."""
        values = {}
        for name, value in self.as_dict().items():
            lo, hi = SAFE_RANGE_NORM[name]
            values[name] = min(max(value, lo), hi)
        return MotorVector(**values)


def _lerp(norm: float, lo: float, hi: float) -> float:
    return lo + norm * (hi - lo)


def to_mujoco_angles(motors: MotorVector) -> dict[str, float]:
    """Normalized motor vector -> MuJoCo joint angles (radians).

    Always clamps to SAFE_RANGE_NORM first, so a bad model output can never
    drive the sim (or, downstream, the real arm) past the safe sub-range.
    """
    safe = motors.clamped()
    return {
        name: _lerp(value, *MUJOCO_ANGLE_RANGE[name])
        for name, value in safe.as_dict().items()
    }


def to_scs_positions(motors: MotorVector) -> dict[int, int]:
    """Normalized motor vector -> {servo_id: goal_position 0-1023}.

    Always clamps to SAFE_RANGE_NORM first -- this is the last line of
    defense before a value is written to a physical servo.
    """
    safe = motors.clamped()
    positions = {}
    for name, value in safe.as_dict().items():
        lo, hi = SCS_POSITION_RANGE[name]
        pos = int(round(_lerp(value, lo, hi)))
        positions[SERVO_ID[name]] = min(max(pos, SCS_MIN), SCS_MAX)
    return positions
