"""
Pick and Place Environment
==========================
A simple 2D grid-world where an agent must:
  1. Navigate to an object
  2. Pick up the object
  3. Navigate to the target location
  4. Place the object at the target location

This environment is designed to be used with the A2C RPE (dopamine) model.

Observation (9 floats, all normalised to [0, 1]):
    agent_x, agent_y, object_x, object_y, holding, target_x, target_y,
    cue_active, task_started

Actions (Discrete 7):
    0 – move up
    1 – move down
    2 – move left
    3 – move right
    4 – pick
    5 – place
    6 – start (valid only after cue appears)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional


class PickAndPlaceEnv(gym.Env):
    """2-D grid-world pick-and-place task compatible with the Gymnasium API."""

    metadata = {"render_modes": ["human", "ansi"]}

    # Action indices
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    PICK = 4
    PLACE = 5
    START = 6

    # Reward shaping
    REWARD_STEP = -0.01
    REWARD_PICK = 2.0
    REWARD_PLACE = 20.0
    REWARD_INVALID = -0.02
    REWARD_START = 0.2

    def __init__(
        self,
        grid_size: int = 5,
        max_steps: int = 200,
        seed: int = None,
        shaping_start: float = 1.0,
        shaping_end: float = 0.5,
        start_cue_max_delay: int = 3,
    ):
        super().__init__()

        if grid_size < 2:
            raise ValueError("grid_size must be at least 2.")

        self.grid_size = grid_size
        self.max_steps = max_steps
        self.start_cue_max_delay = max(0, int(start_cue_max_delay))
        self.shaping_start = shaping_start
        self.shaping_end = shaping_end
        self.shaping_scale = shaping_start
        self._np_rng = np.random.default_rng(seed)

        self.action_space = spaces.Discrete(7)
        # (agent_x, agent_y, obj_x, obj_y, holding, target_x, target_y, cue_active, task_started)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(9,), dtype=np.float32
        )

        # Will be initialised in reset()
        self.agent_pos: np.ndarray = None
        self.object_pos: np.ndarray = None
        self.target_pos: np.ndarray = None
        self.holding: bool = False
        self.object_placed: bool = False
        self._step_count: int = 0
        self.cue_step: int = 1
        self.task_started: bool = False
        self.task_started_step: Optional[int] = None

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, seed: int = None, options: dict = None):
        super().reset(seed=seed)
        if seed is not None:
            self._np_rng = np.random.default_rng(seed)

        n_cells = self.grid_size * self.grid_size
        indices = self._np_rng.choice(n_cells, size=3, replace=False)

        self.agent_pos = np.array(
            [indices[0] // self.grid_size, indices[0] % self.grid_size], dtype=np.int32
        )
        self.object_pos = np.array(
            [indices[1] // self.grid_size, indices[1] % self.grid_size], dtype=np.int32
        )
        self.target_pos = np.array(
            [indices[2] // self.grid_size, indices[2] % self.grid_size], dtype=np.int32
        )
        self.holding = False
        self.object_placed = False
        self._step_count = 0
        self.cue_step = int(self._np_rng.integers(1, self.start_cue_max_delay + 2))
        self.task_started = False
        self.task_started_step = None

        return self._get_obs(), {}

    def set_curriculum(self, progress: float) -> None:
        """
        Set reward-shaping strength by training progress in [0, 1].

        Early episodes use stronger shaping to encourage exploration and
        task decomposition; shaping is annealed later so the policy relies
        more on sparse task rewards.
        """
        p = float(np.clip(progress, 0.0, 1.0))
        self.shaping_scale = self.shaping_start + (self.shaping_end - self.shaping_start) * p

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        self._step_count += 1
        reward = self.REWARD_STEP
        terminated = False
        truncated = False

        cue_active = self._step_count >= self.cue_step

        if not cue_active:
            # Task is locked until cue appears.
            reward = self.REWARD_INVALID
        elif not self.task_started:
            # After cue appears, model must explicitly start task.
            if action == self.START:
                self.task_started = True
                self.task_started_step = self._step_count
                reward = self.REWARD_START
            else:
                reward = self.REWARD_INVALID
        elif action == self.START:
            # START action has no effect once task is underway.
            reward = self.REWARD_INVALID

        elif action < self.PICK:
            prev_goal_dist = self._goal_distance()
            # Movement
            delta = {
                self.UP:    np.array([-1,  0], dtype=np.int32),
                self.DOWN:  np.array([ 1,  0], dtype=np.int32),
                self.LEFT:  np.array([ 0, -1], dtype=np.int32),
                self.RIGHT: np.array([ 0,  1], dtype=np.int32),
            }[action]
            new_pos = np.clip(self.agent_pos + delta, 0, self.grid_size - 1)
            self.agent_pos = new_pos

            curr_goal_dist = self._goal_distance()
            distance_gain = prev_goal_dist - curr_goal_dist
            max_manhattan = max(2 * (self.grid_size - 1), 1)
            reward += self.shaping_scale * (distance_gain / max_manhattan)

        elif action == self.PICK:
            if (
                not self.holding
                and not self.object_placed
                and np.array_equal(self.agent_pos, self.object_pos)
            ):
                self.holding = True
                reward = self.REWARD_PICK
            else:
                reward = self.REWARD_INVALID

        elif action == self.PLACE:
            if self.holding and np.array_equal(self.agent_pos, self.target_pos):
                self.holding = False
                self.object_placed = True
                reward = self.REWARD_PLACE
                terminated = True
            else:
                reward = self.REWARD_INVALID

        if self._step_count >= self.max_steps:
            truncated = True

        obs = self._get_obs()
        info = {
            "holding": self.holding,
            "object_placed": self.object_placed,
            "step_count": self._step_count,
            "cue_active": cue_active,
            "cue_step": self.cue_step,
            "task_started": self.task_started,
            "task_started_step": self.task_started_step,
        }
        return obs, float(reward), terminated, truncated, info

    def render(self, mode: str = "human") -> str:
        """Return an ASCII rendering of the current grid state."""
        g = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        tr, tc = int(self.target_pos[0]), int(self.target_pos[1])
        g[tr][tc] = "T"

        if not self.holding and not self.object_placed:
            or_, oc = int(self.object_pos[0]), int(self.object_pos[1])
            g[or_][oc] = "O"

        ar, ac = int(self.agent_pos[0]), int(self.agent_pos[1])
        # Agent on top of target
        if np.array_equal(self.agent_pos, self.target_pos):
            g[ar][ac] = "AT" if not self.holding else "AT+"
        elif self.holding:
            g[ar][ac] = "A+"
        else:
            g[ar][ac] = "A"

        sep = "+" + ("-" * (self.grid_size * 4 - 1)) + "+"
        lines = [sep]
        for row in g:
            lines.append("| " + " | ".join(f"{c:2}" for c in row) + " |")
            lines.append(sep)

        status = (
            f"Step: {self._step_count}/{self.max_steps}  "
            f"Holding: {self.holding}  "
            f"Placed: {self.object_placed}"
        )
        lines.append(status)
        output = "\n".join(lines)

        if mode == "human":
            print(output)
        return output

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        n = max(self.grid_size - 1, 1)

        if self.holding:
            obj_r, obj_c = self.agent_pos[0], self.agent_pos[1]
        elif self.object_placed:
            obj_r, obj_c = self.target_pos[0], self.target_pos[1]
        else:
            obj_r, obj_c = self.object_pos[0], self.object_pos[1]

        return np.array(
            [
                self.agent_pos[0] / n,
                self.agent_pos[1] / n,
                obj_r / n,
                obj_c / n,
                float(self.holding),
                self.target_pos[0] / n,
                self.target_pos[1] / n,
                float(self._step_count >= self.cue_step),
                float(self.task_started),
            ],
            dtype=np.float32,
        )

    def _goal_distance(self) -> int:
        """Manhattan distance to the current sub-goal (object or target)."""
        if self.holding:
            goal = self.target_pos
        elif self.object_placed:
            goal = self.target_pos
        else:
            goal = self.object_pos
        return int(np.abs(self.agent_pos - goal).sum())
