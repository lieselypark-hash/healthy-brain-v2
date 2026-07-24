"""
Training script for the A2C + RPE (Dopamine) Pick-and-Place agent.

Usage
-----
    python train.py                           # default settings
    python train.py --n_episodes 3000 --lr 3e-4 --grid_size 6
    python train.py --n_episodes 500 --no_save   # quick smoke-test

Key parameters
--------------
--n_episodes    Total number of training episodes.
--n_steps       Number of environment steps before each network update.
--grid_size     Side-length of the square pick-and-place grid.
--hidden_dim    Width of the shared hidden layers.
--lr            Adam learning rate.
--gamma         Discount factor.
--entropy_coef  Entropy bonus weight (exploration).
--value_coef    Critic loss weight.
--log_interval  How many episodes between console logs.
--save_interval How many episodes between checkpoint saves.
--save_dir      Directory for checkpoint files.
--no_save       Disable saving checkpoints.
--seed          Random seed for reproducibility.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from a2c_rpe_model import A2CAgent
from pick_and_place_env import PickAndPlaceEnv


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train A2C with RPE (dopamine) on the Pick-and-Place task."
    )
    parser.add_argument("--n_episodes",  type=int,   default=2000)
    parser.add_argument("--n_steps",     type=int,   default=8,
                        help="Number of steps per A2C update.")
    parser.add_argument("--grid_size",   type=int,   default=5)
    parser.add_argument("--hidden_dim",  type=int,   default=128)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--gamma",       type=float, default=0.99)
    parser.add_argument("--entropy_coef",type=float, default=0.01)
    parser.add_argument("--value_coef",  type=float, default=0.5)
    parser.add_argument("--alpha_tonic", type=float, default=0.005,
                        help="Tonic dopamine EMA coefficient.")
    parser.add_argument("--log_interval",   type=int, default=100)
    parser.add_argument("--save_interval",  type=int, default=500)
    parser.add_argument("--save_dir",       type=str, default="checkpoints")
    parser.add_argument("--no_save",        action="store_true")
    parser.add_argument("--seed",           type=int, default=42)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> tuple[A2CAgent, list, list]:
    """
    Train the A2C agent and return (agent, episode_rewards, episode_lengths).

    The training loop follows the standard n-step A2C pattern:
      1. Collect ``n_steps`` transitions (or until episode end).
      2. Compute RPE for each transition.
      3. Update actor and critic.
      4. Repeat until episode terminates.
    """
    np.random.seed(args.seed)

    env = PickAndPlaceEnv(
        grid_size=args.grid_size,
        max_steps=200,
        seed=args.seed,
    )
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = A2CAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        alpha_tonic=args.alpha_tonic,
    )

    if not args.no_save:
        os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 60)
    print("  A2C + RPE Dopamine Model – Pick-and-Place Task")
    print("=" * 60)
    print(f"  Grid size  : {args.grid_size}×{args.grid_size}")
    print(f"  State dim  : {state_dim}   Action dim: {action_dim}")
    print(f"  Hidden dim : {args.hidden_dim}")
    print(f"  LR={args.lr}  γ={args.gamma}  n_steps={args.n_steps}")
    print("=" * 60)

    episode_rewards: list[float] = []
    episode_lengths: list[int]   = []
    success_count = 0

    for episode in range(args.n_episodes):
        obs, _ = env.reset(seed=args.seed + episode)
        ep_reward = 0.0
        ep_length = 0
        terminated = truncated = False
        last_info: dict = {}

        # Collect experience and update in n-step chunks
        while not (terminated or truncated):
            batch_s, batch_a, batch_r, batch_ns, batch_d = [], [], [], [], []

            for _ in range(args.n_steps):
                action, _ = agent.select_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)

                batch_s.append(obs)
                batch_a.append(action)
                batch_r.append(reward)
                batch_ns.append(next_obs)
                batch_d.append(float(terminated or truncated))

                ep_reward += reward
                ep_length += 1
                obs = next_obs
                last_info = info

                if terminated or truncated:
                    break

            agent.update(batch_s, batch_a, batch_r, batch_ns, batch_d)

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)
        agent.episode_rewards.append(ep_reward)
        agent.episode_lengths.append(ep_length)

        if last_info.get("object_placed", False):
            success_count += 1

        # Logging
        if (episode + 1) % args.log_interval == 0:
            window = slice(max(0, episode + 1 - args.log_interval), episode + 1)
            avg_r  = np.mean(episode_rewards[window])
            avg_l  = np.mean(episode_lengths[window])
            s_rate = success_count / (episode + 1)
            da     = agent.dopamine.get_stats()

            print(
                f"Ep {episode+1:>5}/{args.n_episodes}  "
                f"AvgReward: {avg_r:+7.3f}  "
                f"AvgLen: {avg_l:6.1f}  "
                f"Success: {s_rate:.3f}  "
                f"RPE(mean): {da['mean_rpe']:+.4f}  "
                f"Tonic DA: {da['tonic_level']:+.4f}"
            )

        # Checkpoint
        if not args.no_save and (episode + 1) % args.save_interval == 0:
            ckpt_path = os.path.join(
                args.save_dir, f"a2c_rpe_ep{episode+1}.pt"
            )
            agent.save(ckpt_path)
            print(f"  [checkpoint saved → {ckpt_path}]")

    print()
    print("=" * 60)
    print(f"  Training complete. Episodes: {args.n_episodes}")
    print(f"  Overall success rate: {success_count / args.n_episodes:.3f}")
    da = agent.dopamine.get_stats()
    print(f"  Final tonic dopamine level: {da['tonic_level']:.4f}")
    print("=" * 60)

    if not args.no_save:
        final_path = os.path.join(args.save_dir, "a2c_rpe_final.pt")
        agent.save(final_path)
        print(f"  Final model saved → {final_path}")

    return agent, episode_rewards, episode_lengths


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    train(args)
