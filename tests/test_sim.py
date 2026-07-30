"""
Tests for arm.sim (Layer 2). These build and step the MuJoCo model headlessly
-- they never call ArmSimulator.run(), which opens a real GUI window and
requires mjpython on macOS.
"""

import pytest

mujoco = pytest.importorskip("mujoco")

from arm.motor_contract import MUJOCO_ANGLE_RANGE, SAFE_RANGE_NORM, MotorVector
from arm.sim import CLAW_CLOSED_THRESHOLD, TABLE_Z, TREMOR_HZ, ArmSimulator


def _expected_angle(joint: str, norm: float) -> float:
    lo_safe, hi_safe = SAFE_RANGE_NORM[joint]
    clamped = min(max(norm, lo_safe), hi_safe)
    lo, hi = MUJOCO_ANGLE_RANGE[joint]
    return lo + clamped * (hi - lo)


def test_single_arm_scene_builds_and_applies():
    sim = ArmSimulator(labels=("healthy",))
    sim.set_trajectory("healthy", [MotorVector(0.0, 0.0, 0.0)])
    sim._apply_all()

    track = sim._tracks["healthy"]
    base_adr = track.joint_qpos_adr["base"][0]
    shoulder_adr = track.joint_qpos_adr["shoulder"][0]
    assert sim.data.qpos[base_adr] == pytest.approx(_expected_angle("base", 0.0))
    assert sim.data.qpos[shoulder_adr] == pytest.approx(_expected_angle("shoulder", 0.0))


def test_dual_arm_scene_keeps_arms_independent():
    sim = ArmSimulator(labels=("healthy", "parkinsons"))
    sim.set_trajectory("healthy", [MotorVector(0.0, 0.0, 0.0)])
    sim.set_trajectory("parkinsons", [MotorVector(1.0, 1.0, 1.0)])
    sim._apply_all()

    healthy_base_adr = sim._tracks["healthy"].joint_qpos_adr["base"][0]
    parkinsons_base_adr = sim._tracks["parkinsons"].joint_qpos_adr["base"][0]
    assert sim.data.qpos[healthy_base_adr] == pytest.approx(_expected_angle("base", 0.0))
    assert sim.data.qpos[parkinsons_base_adr] == pytest.approx(_expected_angle("base", 1.0))
    assert sim.data.qpos[healthy_base_adr] != sim.data.qpos[parkinsons_base_adr]


def test_toggle_visible_zeroes_alpha_channel():
    sim = ArmSimulator(labels=("healthy", "parkinsons"))
    geom_id = sim._tracks["healthy"].geom_ids[0]
    assert sim.model.geom_rgba[geom_id][3] == pytest.approx(1.0)

    sim.toggle_visible("healthy")
    assert sim.model.geom_rgba[geom_id][3] == pytest.approx(0.0)

    sim.toggle_visible("healthy")
    assert sim.model.geom_rgba[geom_id][3] == pytest.approx(1.0)


def test_request_new_episode_uses_callback():
    calls = []

    def on_request_episode(label, seed):
        calls.append((label, seed))
        return [MotorVector(0.2, 0.2, 0.2), MotorVector(0.8, 0.8, 0.8)]

    sim = ArmSimulator(labels=("healthy",), on_request_episode=on_request_episode)
    sim.request_new_episode()

    assert calls == [("healthy", 1)]
    assert len(sim._tracks["healthy"].trajectory) == 2


def test_out_of_range_motor_values_never_reach_extreme_servo_travel():
    """Sanity check that clamping is actually applied inside the sim's own apply path,
    not just in the contract module in isolation."""
    sim = ArmSimulator(labels=("healthy",))
    sim.set_trajectory("healthy", [MotorVector(0.0, 0.0, 0.0)])
    sim._apply_all()

    base_adr = sim._tracks["healthy"].joint_qpos_adr["base"][0]
    full_lo, _ = MUJOCO_ANGLE_RANGE["base"]
    # Safe range starts at 0.05, not 0.0, so we should never see the raw extreme.
    assert sim.data.qpos[base_adr] > full_lo


# -- pick / place object visualization ---------------------------------------


def test_object_stays_put_while_claw_is_open():
    sim = ArmSimulator(labels=("healthy",))
    home_pos = sim.data.mocap_pos[sim._tracks["healthy"].object_mocap_id].copy()

    sim.set_trajectory(
        "healthy",
        [MotorVector(0.5, 0.5, 0.0), MotorVector(0.6, 0.6, 0.0)],
    )
    sim._apply_all()

    assert sim.data.mocap_pos[sim._tracks["healthy"].object_mocap_id] == pytest.approx(
        home_pos
    )


def test_object_attaches_to_gripper_once_claw_closes():
    sim = ArmSimulator(labels=("healthy",))
    track = sim._tracks["healthy"]

    sim.set_trajectory("healthy", [MotorVector(0.5, 0.5, 1.0)])
    sim._apply_all()

    assert track.last_motors.claw >= CLAW_CLOSED_THRESHOLD
    wrist_pos = sim.data.xpos[track.wrist_body_id]
    assert sim.data.mocap_pos[track.object_mocap_id] == pytest.approx(wrist_pos)


def test_object_drops_to_table_height_after_release():
    sim = ArmSimulator(labels=("healthy",))
    track = sim._tracks["healthy"]

    # Close the claw at one arm pose (object attaches here, following the
    # gripper -- its Z should be elevated, matching the wrist).
    sim.set_trajectory("healthy", [MotorVector(0.3, 0.3, 1.0)])
    sim._apply_all()
    held_pos = sim.data.mocap_pos[track.object_mocap_id].copy()
    assert held_pos[2] == pytest.approx(sim.data.xpos[track.wrist_body_id][2])

    # ...then move the arm elsewhere and open the claw (release/place).
    sim.set_trajectory("healthy", [MotorVector(0.8, 0.8, 0.0)])
    sim._apply_all()
    released_pos = sim.data.mocap_pos[track.object_mocap_id]

    # Keeps the XY it was released at, but drops to the table -- it must
    # never be left floating at the (elevated) gripper height.
    assert released_pos[0] == pytest.approx(held_pos[0])
    assert released_pos[1] == pytest.approx(held_pos[1])
    assert released_pos[2] == pytest.approx(TABLE_Z)


def test_object_rests_on_table_before_pickup_not_at_gripper_height():
    """Regression test: the object must sit on the table pre-pickup, not
    float at the (possibly elevated) height the gripper will arrive at."""
    sim = ArmSimulator(labels=("healthy",))
    track = sim._tracks["healthy"]

    # A pick pose with the shoulder lifted well off the table.
    sim.set_trajectory("healthy", [MotorVector(0.5, 0.9, 1.0), MotorVector(0.5, 0.9, 1.0)])

    pos_before_playback = sim.data.mocap_pos[track.object_mocap_id]
    assert pos_before_playback[2] == pytest.approx(TABLE_Z)


# -- Parkinsonian tremor (cosmetic rendering effect only) -----------------------


def test_healthy_track_has_no_tremor():
    sim = ArmSimulator(labels=("healthy",), enable_tremor=True)
    sim.set_trajectory("healthy", [MotorVector(0.5, 0.5, 0.0)])
    base_adr = sim._tracks["healthy"].joint_qpos_adr["base"][0]

    angles = set()
    for _ in range(5):
        sim._clock += 0.05
        sim._apply_all()
        angles.add(round(float(sim.data.qpos[base_adr]), 6))

    assert len(angles) == 1  # never moves on its own


def test_parkinsons_track_jitters_over_time_when_tremor_enabled():
    sim = ArmSimulator(labels=("parkinsons",), enable_tremor=True)
    sim.set_trajectory("parkinsons", [MotorVector(0.5, 0.5, 0.0)])
    base_adr = sim._tracks["parkinsons"].joint_qpos_adr["base"][0]

    angles = set()
    steps_per_cycle = int(round(1.0 / TREMOR_HZ / (1.0 / 50.0)))
    for _ in range(steps_per_cycle + 2):
        sim._clock += 1.0 / 50.0
        sim._apply_all()
        angles.add(round(float(sim.data.qpos[base_adr]), 6))

    assert len(angles) > 1  # visibly jitters over one tremor cycle


def test_tremor_amplitude_is_small_not_explosive():
    """The jitter should be a subtle shake, not a wild swing."""
    sim = ArmSimulator(labels=("parkinsons",), enable_tremor=True)
    sim.set_trajectory("parkinsons", [MotorVector(0.5, 0.5, 0.0)])
    base_adr = sim._tracks["parkinsons"].joint_qpos_adr["base"][0]
    baseline = float(sim.data.qpos[base_adr])

    max_deviation = 0.0
    steps_per_cycle = int(round(1.0 / TREMOR_HZ / (1.0 / 50.0)))
    for _ in range(steps_per_cycle * 3):
        sim._clock += 1.0 / 50.0
        sim._apply_all()
        max_deviation = max(max_deviation, abs(float(sim.data.qpos[base_adr]) - baseline))

    assert 0.0 < max_deviation <= 0.03  # small: a shake, not a swing


def test_tremor_disabled_gives_stationary_parkinsons_track():
    sim = ArmSimulator(labels=("parkinsons",), enable_tremor=False)
    sim.set_trajectory("parkinsons", [MotorVector(0.5, 0.5, 0.0)])
    base_adr = sim._tracks["parkinsons"].joint_qpos_adr["base"][0]

    angles = set()
    for _ in range(5):
        sim._clock += 0.05
        sim._apply_all()
        angles.add(round(float(sim.data.qpos[base_adr]), 6))

    assert len(angles) == 1


# -- auto-replay on episode completion -------------------------------------------


def test_finished_episode_auto_replays_after_hold_period():
    calls = []

    def on_request_episode(label, seed):
        calls.append(seed)
        return [MotorVector(0.5, 0.5, 0.0), MotorVector(0.5, 0.5, 0.0)]

    sim = ArmSimulator(
        labels=("healthy",),
        on_request_episode=on_request_episode,
        auto_replay=True,
        replay_hold_seconds=1.0,
    )
    sim.set_trajectory("healthy", [MotorVector(0.5, 0.5, 0.0), MotorVector(0.5, 0.5, 0.0)])

    # Drive `_advance` until the 2-step trajectory (cursor 0->1, then held at 1
    # for another full interpolation window) finishes the episode.
    from arm.sim import SUBSTEPS_PER_STEP

    for _ in range(2 * SUBSTEPS_PER_STEP + 1):
        sim._advance()
    assert sim._tracks["healthy"].finished_announced

    # Not enough hold time has passed yet -- no replay.
    sim._maybe_auto_replay()
    assert calls == []

    # Advance the clock past the hold period and check again.
    sim._clock += 1.5
    sim._maybe_auto_replay()
    assert calls == [1]


def test_auto_replay_disabled_never_calls_back():
    calls = []

    def on_request_episode(label, seed):
        calls.append(seed)
        return [MotorVector(0.5, 0.5, 0.0)]

    sim = ArmSimulator(
        labels=("healthy",), on_request_episode=on_request_episode, auto_replay=False
    )
    sim.set_trajectory("healthy", [MotorVector(0.5, 0.5, 0.0), MotorVector(0.5, 0.5, 0.0)])
    from arm.sim import SUBSTEPS_PER_STEP

    for _ in range(2 * SUBSTEPS_PER_STEP + 1):
        sim._advance()
    sim._clock += 10.0
    sim._maybe_auto_replay()

    assert calls == []
