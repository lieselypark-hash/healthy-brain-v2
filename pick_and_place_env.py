"""
Pick and Place Environment
==========================
A simple 2D grid-world where an agent must:
  1. Navigate to an object
  2. Pick up the object
  3. Navigate to the target location
  4. Place the object at the target location

Observation (9 floats, all normalised to [0, 1]):
    agent_x, agent_y, object_x, object_y, holding, target_x, target_y,
    cue_active, task_started

Actions (Discrete 7):
    0 - move up
    1 - move down
    2 - move left
    3 - move right
    4 - pick
    5 - place
    6 - start (valid only after cue appears)
"""

from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


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

    # Rewards
    REWARD_STEP = -0.01
    REWARD_PICK = 2.0
    REWARD_PLACE = 20.0
    REWARD_INVALID = -0.02
    REWARD_START = 0.0

    def __init__(
        self,
        grid_size: int = 5,
        max_steps: int = 200,
        seed: Optional[int] = None,
        shaping_start: float = 1.0,
        shaping_end: float = 0.5,
        start_cue_min_delay: int = 1,
        start_cue_max_delay: int = 3,
        start_window: Optional[int] = None,
    ):
        super().__init__()

        if grid_size < 2:
            raise ValueError("grid_size must be at least 2.")

        self.grid_size = int(grid_size)
        self.max_steps = int(max_steps)
        self.start_cue_min_delay = max(1, int(start_cue_min_delay))
        self.start_cue_max_delay = max(self.start_cue_min_delay, int(start_cue_max_delay))

        self.shaping_start = float(shaping_start)
        self.shaping_end = float(shaping_end)
        self.shaping_scale = float(shaping_start)

        self.start_window = None if start_window is None else max(1, int(start_window))
        self.fail_on_missed_start = False

        self.reward_step = float(self.REWARD_STEP)
        self.reward_pick = float(self.REWARD_PICK)
        self.reward_place = float(self.REWARD_PLACE)
        self.reward_invalid = float(self.REWARD_INVALID)
        self.reward_start = float(self.REWARD_START)

        self._np_rng = np.random.default_rng(seed)

        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(9,),
            dtype=np.float32,
        )

        # State set by reset()
        self.agent_pos: Optional[np.ndarray] = None
        self.object_pos: Optional[np.ndarray] = None
        self.target_pos: Optional[np.ndarray] = None
        self.holding: bool = False
        self.object_placed: bool = False
        self._step_count: int = 0
        self.cue_step: int = self.start_cue_max_delay
        self.task_started: bool = False
        self.task_started_step: Optional[int] = None
        self.missed_start_window: bool = False

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
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
        self.task_started = False
        self.task_started_step = None
        self.missed_start_window = False

        return self._get_obs(), {}

    def set_curriculum(self, progress: float) -> None:
        """Adjust shaping and cue delay by progress in [0, 1]."""
        p = float(np.clip(progress, 0.0, 1.0))
        self.shaping_scale = self.shaping_start + (self.shaping_end - self.shaping_start) * p
        cue_delay = self.start_cue_min_delay + (
            self.start_cue_max_delay - self.start_cue_min_delay
        ) * p
        self.cue_step = int(round(cue_delay))

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        self._step_count += 1
        reward = self.reward_step
        terminated = False
        truncated = False

        cue_active = self._step_count >= self.cue_step
        missed_start_window = False

        if not cue_active:
            reward = self.reward_invalid
        elif not self.task_started:
            if action == self.START and self._start_window_open():
                self.task_started = True
                self.task_started_step = self._step_count
                reward = self.reward_start
            else:
                reward = self.reward_invalid
                if self._start_window_expired_after_action():
                    missed_start_window = True
                    self.missed_start_window = True
                    terminated = self.fail_on_missed_start
        elif action == self.START:
            reward = self.reward_invalid
        elif action < self.PICK:
            prev_goal_dist = self._goal_distance()
            delta = {
                self.UP: np.array([-1, 0], dtype=np.int32),
                self.DOWN: np.array([1, 0], dtype=np.int32),
                self.LEFT: np.array([0, -1], dtype=np.int32),
                self.RIGHT: np.array([0, 1], dtype=np.int32),
            }[action]
            self.agent_pos = np.clip(self.agent_pos + delta, 0, self.grid_size - 1)

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
                reward = self.reward_pick
            else:
                reward = self.reward_invalid
        elif action == self.PLACE:
            if self.holding and np.array_equal(self.agent_pos, self.target_pos):
                self.holding = False
                self.object_placed = True
                reward = self.reward_place
                terminated = True
            else:
                reward = self.reward_invalid

        if self._step_count >= self.max_steps:
            truncated = True

        info = {
            "holding": self.holding,
            "object_placed": self.object_placed,
            "step_count": self._step_count,
            "cue_active": cue_active,
            "cue_step": self.cue_step,
            "task_started": self.task_started,
            "task_started_step": self.task_started_step,
            "start_window": self.start_window,
            "missed_start_window": missed_start_window,
        }
        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self, mode: str = "human") -> str:
        """Return an ASCII rendering of the current grid state."""
        g = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        tr, tc = int(self.target_pos[0]), int(self.target_pos[1])
        g[tr][tc] = "T"

        if not self.holding and not self.object_placed:
            or_, oc = int(self.object_pos[0]), int(self.object_pos[1])
            g[or_][oc] = "O"

        ar, ac = int(self.agent_pos[0]), int(self.agent_pos[1])
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

        lines.append(
            f"Step: {self._step_count}/{self.max_steps}  Holding: {self.holding}  Placed: {self.object_placed}"
        )
        output = "\n".join(lines)

        if mode == "human":
            print(output)
        return output

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
        if self.holding or self.object_placed:
            goal = self.target_pos
        else:
            goal = self.object_pos
        return int(np.abs(self.agent_pos - goal).sum())

    def _start_window_deadline(self) -> Optional[int]:
        if self.start_window is None:
            return None
        return self.cue_step + self.start_window - 1

    def _start_window_open(self) -> bool:
        deadline = self._start_window_deadline()
        if deadline is None:
            return True
        return self._step_count <= deadline

    def _start_window_expired_after_action(self) -> bool:
        deadline = self._start_window_deadline()
        if deadline is None:
            return False
        return self._step_count >= deadline
