"""
Evaluation script for a trained A2C + RPE pick-and-place agent.

Usage
-----
    python evaluate.py                                 # random policy (no checkpoint)
    python evaluate.py --checkpoint checkpoints/a2c_rpe_final.pt
    python evaluate.py --checkpoint checkpoints/a2c_rpe_final.pt --render --episodes 5
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from a2c_rpe_model import A2CAgent
from pick_and_place_env import PickAndPlaceEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained A2C RPE agent on the Pick-and-Place task."
    )
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a .pt checkpoint file.")
    parser.add_argument("--grid_size",  type=int, default=5)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--episodes",   type=int, default=20)
    parser.add_argument("--render",     action="store_true",
                        help="Print ASCII grid after each step.")
    parser.add_argument("--seed",       type=int, default=0)
    return parser.parse_args()


def evaluate(args: argparse.Namespace) -> dict:
    """Run the agent for ``args.episodes`` episodes and return summary stats."""
    env = PickAndPlaceEnv(grid_size=args.grid_size, max_steps=200)
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = A2CAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
    )

    if args.checkpoint:
        agent.load(args.checkpoint)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("No checkpoint provided – using randomly initialised weights.")

    rewards, lengths, successes = [], [], []

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        ep_reward = 0.0
        ep_length = 0
        terminated = truncated = False

        while not (terminated or truncated):
            if args.render:
                env.render()

            action, _ = agent.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_length += 1

        if args.render:
            env.render()
            print(f"--- Episode {ep+1} end. Reward={ep_reward:.2f} ---\n")

        rewards.append(ep_reward)
        lengths.append(ep_length)
        successes.append(info.get("object_placed", False))

    stats = {
        "mean_reward": float(np.mean(rewards)),
        "std_reward":  float(np.std(rewards)),
        "mean_length": float(np.mean(lengths)),
        "success_rate": float(np.mean(successes)),
        "episodes": args.episodes,
    }

    print(f"\nEvaluation over {args.episodes} episodes:")
    print(f"  Mean reward  : {stats['mean_reward']:.3f} ± {stats['std_reward']:.3f}")
    print(f"  Mean length  : {stats['mean_length']:.1f}")
    print(f"  Success rate : {stats['success_rate']:.3f}")
    return stats


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
