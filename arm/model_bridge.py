"""
Layer 1 -- model output
=========================
Adapts the existing grid-world RL agents (a2c_rpe_model / parkinsons_a2c_rpe_model)
to the motor-value contract defined in arm.motor_contract, WITHOUT touching the
original training code, checkpoints, or graph generation.

Why this mapping
-----------------
The agents here don't output continuous joint values -- they choose among 7
discrete grid-navigation actions (see pick_and_place_env.py). But the resulting
*observation* already carries normalized [0, 1] state that corresponds closely
to a 3-DOF arm's [base, shoulder, claw]:

    obs = [agent_x, agent_y, obj_x, obj_y, holding, target_x, target_y, cue_active, task_started]

    base     = agent_x   (normalized grid row)
    shoulder = agent_y   (normalized grid column)
    claw     = holding   (0 = open, 1 = closed)

get_motor_values() is therefore a pure function of the observation, independent
of which policy produced it. Running a full episode with the healthy agent vs.
the Parkinson's agent (same seed => same start pose/target) produces two motor
trajectories whose differences -- pauses, stalls, indecisive back-and-forth
movement -- are the visible "motor signature" of the impairment, without
hand-coding impairment logic here. Nothing in this module adds noise or any
other synthetic effect; every value is exactly what the trained policy chose.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from a2c_rpe_model import A2CAgent as HealthyA2CAgent  # noqa: E402
from parkinsons_a2c_rpe_model import A2CAgent as ParkinsonsA2CAgent  # noqa: E402
from pick_and_place_env import PickAndPlaceEnv  # noqa: E402

from arm.motor_contract import MotorVector  # noqa: E402

VARIANTS = ("healthy", "parkinsons")

DEFAULT_CHECKPOINTS = {
    "healthy": os.path.join(_REPO_ROOT, "checkpoints", "a2c_rpe_final.pt"),
    "parkinsons": os.path.join(_REPO_ROOT, "checkpoints", "parkinsons", "a2c_rpe_final.pt"),
}

STATE_DIM = 9
ACTION_DIM = 7


def get_motor_values(obs: np.ndarray) -> MotorVector:
    """Map one PickAndPlaceEnv observation to a normalized [base, shoulder, claw] vector."""
    return MotorVector(base=float(obs[0]), shoulder=float(obs[1]), claw=float(obs[4]))


def load_agent(
    variant: str,
    checkpoint: Optional[str] = None,
    *,
    hidden_dim: int = 128,
    allow_untrained: bool = False,
):
    """Instantiate and load the agent for `variant` ("healthy" or "parkinsons")."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; expected one of {VARIANTS}")

    agent_cls = HealthyA2CAgent if variant == "healthy" else ParkinsonsA2CAgent
    agent = agent_cls(state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden_dim=hidden_dim)

    path = checkpoint or DEFAULT_CHECKPOINTS[variant]
    if path and os.path.exists(path):
        agent.load(path)
    elif not allow_untrained:
        raise FileNotFoundError(
            f"No checkpoint found for variant={variant!r} at {path!r}. "
            "Train one first (see train.py --agent_variant ...), pass an explicit "
            "checkpoint path, or call load_agent(..., allow_untrained=True)."
        )
    return agent


@dataclass
class EpisodeStep:
    step: int
    motors: MotorVector
    obs: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict = field(default_factory=dict)

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


def rollout_episode(
    variant: str,
    checkpoint: Optional[str] = None,
    *,
    seed: int = 0,
    grid_size: int = 5,
    max_steps: int = 200,
    agent=None,
) -> list[EpisodeStep]:
    """Run one full PickAndPlaceEnv episode and return the resulting motor trajectory.

    Pass the same `seed` for two variants to get identical start pose / target,
    for a fair side-by-side comparison.
    """
    if agent is None:
        agent = load_agent(variant, checkpoint)

    env = PickAndPlaceEnv(
        grid_size=grid_size,
        max_steps=max_steps,
        shaping_start=0.0,
        shaping_end=0.0,
    )
    obs, _ = env.reset(seed=seed)

    steps = [
        EpisodeStep(
            step=0,
            motors=get_motor_values(obs),
            obs=obs,
            reward=0.0,
            terminated=False,
            truncated=False,
        )
    ]

    terminated = truncated = False
    step_idx = 0
    while not (terminated or truncated):
        action, _ = agent.select_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        step_idx += 1
        steps.append(
            EpisodeStep(
                step=step_idx,
                motors=get_motor_values(obs),
                obs=obs,
                reward=float(reward),
                terminated=terminated,
                truncated=truncated,
                info=info,
            )
        )
    return steps
