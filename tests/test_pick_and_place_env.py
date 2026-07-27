"""Unit tests for PickAndPlaceEnv."""

import numpy as np
import pytest

from pick_and_place_env import PickAndPlaceEnv


class TestPickAndPlaceEnvInit:
    def test_default_construction(self):
        env = PickAndPlaceEnv()
        assert env.grid_size == 5
        assert env.max_steps == 200
        assert env.action_space.n == 7
        assert env.observation_space.shape == (9,)

    def test_custom_grid_size(self):
        env = PickAndPlaceEnv(grid_size=8)
        assert env.grid_size == 8

    def test_invalid_grid_size_raises(self):
        with pytest.raises(ValueError):
            PickAndPlaceEnv(grid_size=1)

class TestPickAndPlaceEnvReset:
    def test_obs_shape_and_bounds(self):
        env = PickAndPlaceEnv(seed=0)
        obs, _ = env.reset()
        assert obs.shape == (9,)
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
        assert not np.array_equal(env.agent_pos, env.object_pos)
        assert not np.array_equal(env.agent_pos, env.target_pos)
        assert not np.array_equal(env.object_pos, env.target_pos)

    def test_seed_reproducibility(self):
        env1 = PickAndPlaceEnv(seed=99)
        obs1, _ = env1.reset()
        env2 = PickAndPlaceEnv(seed=99)
        obs2, _ = env2.reset()
        np.testing.assert_array_equal(obs1, obs2)

    def test_cue_delay_is_deterministic_without_jitter(self):
        env = PickAndPlaceEnv(seed=13, start_cue_max_delay=4)
        env.set_curriculum(0.0)
        env.reset()
        cue_1 = env.cue_step
        env.reset()
        cue_2 = env.cue_step
        assert cue_1 == cue_2 == 1

    def test_curriculum_makes_cue_easier_early(self):
        env = PickAndPlaceEnv(seed=14, start_cue_max_delay=4)
        env.set_curriculum(0.0)
        env.reset()
        early_cue = env.cue_step
        env.set_curriculum(1.0)
        env.reset()
        late_cue = env.cue_step
        assert early_cue == 1
        assert late_cue == 4
        assert early_cue <= late_cue


class TestPickAndPlaceEnvStep:
    def _start_task(self, env):
        env.cue_step = 1
        env.step(PickAndPlaceEnv.START)

    def test_movement_changes_agent_pos(self):
        env = PickAndPlaceEnv(grid_size=5, seed=10)
        env.reset()
        self._start_task(env)
        env.agent_pos = np.array([2, 2], dtype=np.int32)
        original = env.agent_pos.copy()
        env.step(PickAndPlaceEnv.DOWN)
        assert not np.array_equal(env.agent_pos, original)

    def test_movement_respects_grid_bounds(self):
        env = PickAndPlaceEnv(grid_size=5, seed=10)
        env.reset()
        self._start_task(env)
        env.agent_pos = np.array([0, 0], dtype=np.int32)
        env.step(PickAndPlaceEnv.UP)
        env.step(PickAndPlaceEnv.LEFT)
        assert env.agent_pos[0] == 0
        assert env.agent_pos[1] == 0

    def test_step_returns_valid_obs(self):
        env = PickAndPlaceEnv(seed=5)
        env.reset()
        obs, reward, terminated, truncated, info = env.step(0)
        assert obs.shape == (9,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "holding" in info and "object_placed" in info

    def test_start_required_after_cue(self):
        env = PickAndPlaceEnv(seed=6)
        env.reset()
        env.cue_step = 1
        _, r1, _, _, info1 = env.step(PickAndPlaceEnv.RIGHT)
        assert info1["task_started"] is False
        assert r1 == PickAndPlaceEnv.REWARD_INVALID

        _, r2, _, _, info2 = env.step(PickAndPlaceEnv.START)
        assert info2["task_started"] is True
        assert r2 == env.reward_start

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
        env.cue_step = 1
        env.task_started = True
        env.task_started_step = 1
        return env

    def test_pick_reward_and_holding_flag(self):
        env = self._make_env_with_positions([2, 2], [2, 2], [4, 4])
        _, reward, _, _, info = env.step(PickAndPlaceEnv.PICK)
        assert reward == PickAndPlaceEnv.REWARD_PICK
        assert info["holding"] is True

    def test_place_reward_and_termination(self):
        env = self._make_env_with_positions([3, 3], [3, 3], [3, 3])
        env.step(PickAndPlaceEnv.PICK)
        _, reward, terminated, _, info = env.step(PickAndPlaceEnv.PLACE)
        assert reward == PickAndPlaceEnv.REWARD_PLACE
        assert terminated is True
        assert info["object_placed"] is True

    def test_full_sequence_succeeds(self):
        env = self._make_env_with_positions([0, 0], [0, 2], [0, 4])

        total_reward = 0.0
        for _ in range(2):
            _, r, _, _, _ = env.step(PickAndPlaceEnv.RIGHT)
            total_reward += r

        _, r, _, _, info = env.step(PickAndPlaceEnv.PICK)
        total_reward += r
        assert info["holding"] is True

        for _ in range(2):
            _, r, _, _, _ = env.step(PickAndPlaceEnv.RIGHT)
            total_reward += r

        _, r, terminated, _, info = env.step(PickAndPlaceEnv.PLACE)
        total_reward += r
        assert terminated is True
        assert info["object_placed"] is True
        assert total_reward > 0


class TestPickAndPlaceRender:
    def test_render_returns_string(self):
        env = PickAndPlaceEnv(seed=20)
        env.reset()
        output = env.render(mode="ansi")
        assert isinstance(output, str)
        assert "A" in output
        assert "T" in output
