"""Tests for the motor-value contract (arm.motor_contract)."""

import pytest

from arm.motor_contract import (
    SAFE_RANGE_NORM,
    SCS_POSITION_RANGE,
    MotorVector,
    to_mujoco_angles,
    to_scs_positions,
)


def test_motor_vector_rejects_out_of_range():
    with pytest.raises(ValueError):
        MotorVector(base=1.5, shoulder=0.5, claw=0.5)
    with pytest.raises(ValueError):
        MotorVector(base=0.5, shoulder=-0.1, claw=0.5)


def test_motor_vector_accepts_boundary_values():
    MotorVector(base=0.0, shoulder=0.0, claw=0.0)
    MotorVector(base=1.0, shoulder=1.0, claw=1.0)


def test_clamped_respects_safe_range_per_joint():
    out_of_range = MotorVector(base=0.0, shoulder=1.0, claw=0.5).clamped()
    base_lo, _ = SAFE_RANGE_NORM["base"]
    _, shoulder_hi = SAFE_RANGE_NORM["shoulder"]
    assert out_of_range.base == pytest.approx(base_lo)
    assert out_of_range.shoulder == pytest.approx(shoulder_hi)
    assert out_of_range.claw == pytest.approx(0.5)  # already inside [0, 1]


def test_clamped_leaves_in_range_values_untouched():
    motors = MotorVector(base=0.5, shoulder=0.5, claw=0.5)
    assert motors.clamped().as_tuple() == pytest.approx((0.5, 0.5, 0.5))


def test_to_scs_positions_uses_correct_servo_ids():
    motors = MotorVector(base=0.5, shoulder=0.5, claw=0.5)
    positions = to_scs_positions(motors)
    assert set(positions.keys()) == {1, 2, 3}


def test_to_scs_positions_midpoint():
    motors = MotorVector(base=0.5, shoulder=0.5, claw=0.5)
    positions = to_scs_positions(motors)
    base_lo, base_hi = SCS_POSITION_RANGE["base"]
    assert positions[1] == round((base_lo + base_hi) / 2)


def test_to_scs_positions_clamps_extreme_inputs_within_safe_sub_range():
    low = to_scs_positions(MotorVector(base=0.0, shoulder=0.0, claw=0.0))
    high = to_scs_positions(MotorVector(base=1.0, shoulder=1.0, claw=1.0))

    for name, servo_id in (("base", 1), ("shoulder", 2), ("claw", 3)):
        lo, hi = SCS_POSITION_RANGE[name]
        safe_lo, safe_hi = SAFE_RANGE_NORM[name]
        expected_low = round(lo + safe_lo * (hi - lo))
        expected_high = round(lo + safe_hi * (hi - lo))
        assert low[servo_id] == expected_low
        assert high[servo_id] == expected_high
        # Never touches the raw 0-1023 extremes when the safe range is narrower.
        assert 0 <= low[servo_id] <= 1023
        assert 0 <= high[servo_id] <= 1023


def test_to_mujoco_angles_returns_all_three_joints():
    motors = MotorVector(base=0.2, shoulder=0.8, claw=1.0)
    angles = to_mujoco_angles(motors)
    assert set(angles.keys()) == {"base", "shoulder", "claw"}
    assert all(isinstance(v, float) for v in angles.values())
