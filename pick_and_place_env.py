"""
Pick and Place Environment
==========================
A simple 2D grid-world where an agent must:
  1. Navigate to an object
  2. Pick up the object
  3. Navigate to the target location
  4. Place the object at the target location

This environment is designed to be used with the A2C RPE (dopamine) model.

Observation (7 floats, all normalised to [0, 1]):
    agent_x, agent_y, object_x, object_y, holding, target_x, target_y

Actions (Discrete 6):
    0 – move up
    1 – move down
    2 – move left
    3 – move right
    4 – pick
    5 – place
"""

import numpy as np
import gymnasium as gym
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

    # Reward shaping
    REWARD_STEP = -0.01
    REWARD_PICK = 1.0
    REWARD_PLACE = 10.0
    REWARD_INVALID = -0.1

    def __init__(self, grid_size: int = 5, max_steps: int = 200, seed: int = None):
        super().__init__()

        if grid_size < 2:
            raise ValueError("grid_size must be at least 2.")

        self.grid_size = grid_size
        self.max_steps = max_steps
        self._np_rng = np.random.default_rng(seed)

        self.action_space = spaces.Discrete(6)
        # (agent_x, agent_y, obj_x, obj_y, holding, target_x, target_y) – normalised
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(7,), dtype=np.float32
        )

        # Will be initialised in reset()
        self.agent_pos: np.ndarray = None
        self.object_pos: np.ndarray = None
        self.target_pos: np.ndarray = None
        self.holding: bool = False
        self.object_placed: bool = False
        self._step_count: int = 0

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

        return self._get_obs(), {}

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        self._step_count += 1
        reward = self.REWARD_STEP
        terminated = False
        truncated = False

        if action < self.PICK:
            # Movement
            delta = {
                self.UP:    np.array([-1,  0], dtype=np.int32),
                self.DOWN:  np.array([ 1,  0], dtype=np.int32),
                self.LEFT:  np.array([ 0, -1], dtype=np.int32),
                self.RIGHT: np.array([ 0,  1], dtype=np.int32),
            }[action]
            new_pos = np.clip(self.agent_pos + delta, 0, self.grid_size - 1)
            self.agent_pos = new_pos

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
            ],
            dtype=np.float32,
        )
