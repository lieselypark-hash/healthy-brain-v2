"""
Unit tests for PickAndPlaceEnv.
"""

import numpy as np
import pytest

from pick_and_place_env import PickAndPlaceEnv


class TestPickAndPlaceEnvInit:
    def test_default_construction(self):
        env = PickAndPlaceEnv()
        assert env.grid_size == 5
        assert env.max_steps == 200
        assert env.action_space.n == 6
        assert env.observation_space.shape == (7,)

    def test_custom_grid_size(self):
        env = PickAndPlaceEnv(grid_size=8)
        assert env.grid_size == 8

    def test_invalid_grid_size_raises(self):
        with pytest.raises(ValueError):
            PickAndPlaceEnv(grid_size=1)


class TestPickAndPlaceEnvReset:
    def test_obs_shape_and_bounds(self):
        env = PickAndPlaceEnv(seed=0)
        obs, info = env.reset()
        assert obs.shape == (7,)
        assert np.all(obs >= 0.0) and np.all(obs <= 1.0)
        assert obs.dtype == np.float32

    def test_holding_false_after_reset(self):
        env = PickAndPlaceEnv(seed=1)
        env.reset()
        assert env.holding is False
        assert env.object_placed is False

    def test_positions_differ_after_reset(self):
        env = PickAndPlaceEnv(seed=2)
        env.reset()
        # Agent, object, and target must all be on different cells
        assert not np.array_equal(env.agent_pos, env.object_pos)
        assert not np.array_equal(env.agent_pos, env.target_pos)
        assert not np.array_equal(env.object_pos, env.target_pos)

    def test_seed_reproducibility(self):
        env1 = PickAndPlaceEnv(seed=99)
        obs1, _ = env1.reset()
        env2 = PickAndPlaceEnv(seed=99)
        obs2, _ = env2.reset()
        np.testing.assert_array_equal(obs1, obs2)


class TestPickAndPlaceEnvStep:
    def test_movement_changes_agent_pos(self):
        env = PickAndPlaceEnv(grid_size=5, seed=10)
        env.reset()
        # Force agent to middle so any move is valid
        env.agent_pos = np.array([2, 2], dtype=np.int32)
        original = env.agent_pos.copy()
        env.step(PickAndPlaceEnv.DOWN)   # move down
        assert not np.array_equal(env.agent_pos, original)

    def test_movement_respects_grid_bounds(self):
        env = PickAndPlaceEnv(grid_size=5, seed=10)
        env.reset()
        env.agent_pos = np.array([0, 0], dtype=np.int32)
        env.step(PickAndPlaceEnv.UP)    # already at row 0 – cannot go further up
        env.step(PickAndPlaceEnv.LEFT)  # already at col 0
        assert env.agent_pos[0] == 0
        assert env.agent_pos[1] == 0

    def test_step_returns_valid_obs(self):
        env = PickAndPlaceEnv(seed=5)
        env.reset()
        obs, reward, terminated, truncated, info = env.step(0)
        assert obs.shape == (7,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "holding" in info and "object_placed" in info

    def test_invalid_pick_gives_negative_reward(self):
        env = PickAndPlaceEnv(seed=7)
        env.reset()
        # Move agent away from object
        env.agent_pos = np.array([0, 0], dtype=np.int32)
        env.object_pos = np.array([4, 4], dtype=np.int32)
        env.holding = False
        _, reward, _, _, _ = env.step(PickAndPlaceEnv.PICK)
        assert reward == PickAndPlaceEnv.REWARD_INVALID

    def test_invalid_place_gives_negative_reward(self):
        env = PickAndPlaceEnv(seed=8)
        env.reset()
        env.holding = False  # Not holding anything
        _, reward, _, _, _ = env.step(PickAndPlaceEnv.PLACE)
        assert reward == PickAndPlaceEnv.REWARD_INVALID

    def test_max_steps_triggers_truncated(self):
        env = PickAndPlaceEnv(grid_size=5, max_steps=3, seed=9)
        env.reset()
        for _ in range(3):
            _, _, terminated, truncated, _ = env.step(0)
        assert truncated or terminated

    def test_invalid_action_raises(self):
        env = PickAndPlaceEnv(seed=11)
        env.reset()
        with pytest.raises(AssertionError):
            env.step(99)


class TestPickAndPlaceFullTask:
    def _make_env_with_positions(self, agent, obj, target):
        env = PickAndPlaceEnv(grid_size=5, seed=0)
        env.reset()
        env.agent_pos = np.array(agent, dtype=np.int32)
        env.object_pos = np.array(obj, dtype=np.int32)
        env.target_pos = np.array(target, dtype=np.int32)
        env.holding = False
        env.object_placed = False
        env._step_count = 0
        return env

    def test_pick_reward_and_holding_flag(self):
        env = self._make_env_with_positions([2, 2], [2, 2], [4, 4])
        _, reward, _, _, info = env.step(PickAndPlaceEnv.PICK)
        assert reward == PickAndPlaceEnv.REWARD_PICK
        assert info["holding"] is True

    def test_place_reward_and_termination(self):
        env = self._make_env_with_positions([3, 3], [3, 3], [3, 3])
        # Pick the object first
        env.step(PickAndPlaceEnv.PICK)
        # Now place it (agent is already at target)
        _, reward, terminated, _, info = env.step(PickAndPlaceEnv.PLACE)
        assert reward == PickAndPlaceEnv.REWARD_PLACE
        assert terminated is True
        assert info["object_placed"] is True

    def test_full_sequence_succeeds(self):
        """Navigate to object, pick, navigate to target, place."""
        # Lay out positions so the solution is predictable:
        # agent=(0,0), object=(0,2), target=(0,4) – all in top row
        env = self._make_env_with_positions([0, 0], [0, 2], [0, 4])

        total_reward = 0.0

        # Move right twice to reach object
        for _ in range(2):
            _, r, _, _, _ = env.step(PickAndPlaceEnv.RIGHT)
            total_reward += r

        # Pick
        _, r, _, _, info = env.step(PickAndPlaceEnv.PICK)
        total_reward += r
        assert info["holding"] is True

        # Move right twice to reach target
        for _ in range(2):
            _, r, _, _, _ = env.step(PickAndPlaceEnv.RIGHT)
            total_reward += r

        # Place
        _, r, terminated, _, info = env.step(PickAndPlaceEnv.PLACE)
        total_reward += r
        assert terminated is True
        assert info["object_placed"] is True
        assert total_reward > 0  # Net positive despite step penalties


class TestPickAndPlaceRender:
    def test_render_returns_string(self):
        env = PickAndPlaceEnv(seed=20)
        env.reset()
        output = env.render(mode="ansi")
        assert isinstance(output, str)
        assert "A" in output   # agent marker
        assert "T" in output   # target marker
