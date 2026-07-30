"""Tests for arm.model_bridge (Layer 1: RL model -> motor-vector contract)."""

import os

import numpy as np
import pytest

from arm.model_bridge import (
    DEFAULT_CHECKPOINTS,
    get_motor_values,
    load_agent,
    rollout_episode,
)
from arm.motor_contract import MotorVector


def test_get_motor_values_maps_observation_fields():
    # obs = [agent_x, agent_y, obj_x, obj_y, holding, target_x, target_y, cue_active, task_started]
    obs = np.array([0.25, 0.75, 0.1, 0.1, 1.0, 0.9, 0.9, 1.0, 1.0], dtype=np.float32)
    motors = get_motor_values(obs)
    assert motors == MotorVector(base=0.25, shoulder=0.75, claw=1.0)


def test_load_agent_missing_checkpoint_raises_clear_error():
    with pytest.raises(FileNotFoundError):
        load_agent("healthy", checkpoint="/nonexistent/path.pt")


def test_load_agent_allow_untrained_does_not_require_checkpoint():
    agent = load_agent("healthy", checkpoint="/nonexistent/path.pt", allow_untrained=True)
    assert agent is not None


def test_load_agent_rejects_unknown_variant():
    with pytest.raises(ValueError):
        load_agent("not_a_real_variant")


def test_rollout_episode_produces_valid_motor_vectors_untrained():
    agent = load_agent("parkinsons", checkpoint="/nonexistent/path.pt", allow_untrained=True)
    steps = rollout_episode(
        "parkinsons", agent=agent, seed=0, grid_size=5, max_steps=20
    )
    assert len(steps) >= 2
    for step in steps:
        for value in step.motors.as_tuple():
            assert 0.0 <= value <= 1.0
    assert steps[-1].done


def test_same_seed_gives_identical_start_pose_across_variants():
    healthy_agent = load_agent("healthy", checkpoint="/nonexistent/path.pt", allow_untrained=True)
    parkinsons_agent = load_agent(
        "parkinsons", checkpoint="/nonexistent/path.pt", allow_untrained=True
    )
    healthy_steps = rollout_episode("healthy", agent=healthy_agent, seed=7, max_steps=5)
    parkinsons_steps = rollout_episode("parkinsons", agent=parkinsons_agent, seed=7, max_steps=5)
    assert healthy_steps[0].motors == parkinsons_steps[0].motors


@pytest.mark.parametrize("variant", ["healthy", "parkinsons"])
def test_rollout_with_real_trained_checkpoint_if_present(variant):
    checkpoint = DEFAULT_CHECKPOINTS[variant]
    if not os.path.exists(checkpoint):
        pytest.skip(f"no trained checkpoint at {checkpoint}")
    steps = rollout_episode(variant, seed=0, max_steps=30)
    assert len(steps) >= 2
