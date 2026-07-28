"""
Evaluation script for a trained A2C + RPE pick-and-place agent.

Usage
-----
    python evaluate.py                                 # uses checkpoints/a2c_rpe_final.pt
    python evaluate.py --checkpoint checkpoints/a2c_rpe_final.pt
    python evaluate.py --checkpoint checkpoints/a2c_rpe_final.pt --render --episodes 500
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys


def _ensure_evaluation_runtime() -> None:
    """Re-launch with workspace venv when core dependencies are missing."""
    required = ("numpy", "torch")
    if all(importlib.util.find_spec(name) is not None for name in required):
        return

    if os.environ.get("HB_EVAL_REEXEC") == "1":
        return

    repo_root = os.path.dirname(os.path.abspath(__file__))
    venv_python = os.path.join(repo_root, ".venv", "bin", "python")
    if os.path.exists(venv_python):
        env = os.environ.copy()
        env["HB_EVAL_REEXEC"] = "1"
        os.execve(venv_python, [venv_python, os.path.abspath(__file__), *sys.argv[1:]], env)


_ensure_evaluation_runtime()

import numpy as np
import torch

from a2c_rpe_model import A2CAgent
from pick_and_place_env import PickAndPlaceEnv
from results import generate_plots_from_metrics


def _started_on_time(info: dict, grace_steps: int = 2) -> bool:
    """Return True when START occurs within the cue window plus grace."""
    if not bool(info.get("task_started", False)):
        return False

    start_step = info.get("task_started_step")
    cue_step = info.get("cue_step")
    start_window = info.get("start_window")

    if start_step is None or cue_step is None:
        return bool(info.get("task_started", False))

    if start_window is None:
        deadline = int(cue_step)
    else:
        deadline = int(cue_step) + max(1, int(start_window)) - 1

    deadline += max(0, int(grace_steps))

    return int(start_step) <= deadline


def _is_timed_success(info: dict, episode_length: int, success_time_limit: int) -> bool:
    """Return True only when the task is completed within the step limit."""
    return bool(info.get("object_placed", False)) and int(episode_length) <= int(success_time_limit)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained A2C RPE agent on the Pick-and-Place task."
    )
    parser.add_argument("--checkpoint", type=str, default="checkpoints/a2c_rpe_final.pt",
                        help="Path to a .pt checkpoint file.")
    parser.add_argument(
        "--agent_variant",
        type=str,
        default="normal",
        choices=("normal", "normal_no_shaping"),
        help="Evaluation variant. 'normal_no_shaping' disables movement reward shaping.",
    )
    parser.add_argument("--grid_size",  type=int, default=5)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--episodes",   type=int, default=500)
    parser.add_argument("--success_time_limit", type=int, default=75,
                        help="Count success only when placement occurs within this many steps.")
    parser.add_argument(
        "--start_grace_steps",
        type=int,
        default=2,
        help="Extra steps allowed beyond the strict cue window for counting START as on-time.",
    )
    parser.add_argument("--render",     action="store_true",
                        help="Print ASCII grid after each step.")
    parser.add_argument("--seed",       type=int, default=0)
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Directory to store evaluation metrics and plots.")
    parser.add_argument("--metrics_filename", type=str, default="evaluation_normal_metrics.csv",
                        help="CSV filename for per-episode evaluation metrics.")
    parser.add_argument("--success_plot_filename", type=str, default="normal_success_rate.png",
                        help="Filename for evaluation success-rate plot.")
    parser.add_argument("--reward_plot_filename", type=str, default="normal_reward.png",
                        help="Filename for evaluation reward plot.")
    return parser.parse_args()


def save_evaluation_metrics(path: str, rows: list[dict]) -> None:
    """Save per-episode evaluation metrics to CSV."""
    if not rows:
        return

    fieldnames = [
        "episode",
        "reward",
        "episode_length",
        "completed_task",
        "success",
        "started_task",
        "started_on_time",
        "cumulative_success_rate",
        "rolling_success_rate",
        "cumulative_start_rate",
        "rolling_start_rate",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(args: argparse.Namespace) -> dict:
    """Run the agent for ``args.episodes`` episodes and return summary stats."""
    no_reward_shaping = args.agent_variant.endswith("_no_shaping")
    env_kwargs = {
        "grid_size": args.grid_size,
        "max_steps": 200,
    }
    if no_reward_shaping:
        env_kwargs.update({"shaping_start": 0.0, "shaping_end": 0.0})
    env = PickAndPlaceEnv(**env_kwargs)
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
        print("Evaluation normal variant: " + args.agent_variant)
        print("Reward shaping: " + ("disabled" if no_reward_shaping else "default"))
    else:
        print("No checkpoint provided – using randomly initialised weights.")

    rewards, lengths, successes, starts = [], [], [], []
    rows: list[dict] = []

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
        completed_task = bool(info.get("object_placed", False))
        timed_success = _is_timed_success(info, ep_length, args.success_time_limit)
        successes.append(timed_success)
        starts.append(_started_on_time(info, grace_steps=args.start_grace_steps))

        cumulative_success_rate = float(np.mean(successes))
        cumulative_start_rate = float(np.mean(starts))
        rolling_start_rate = float(np.mean(starts[max(0, len(starts) - 50):]))
        rows.append(
            {
                "episode": ep + 1,
                "reward": float(ep_reward),
                "episode_length": int(ep_length),
                "completed_task": int(completed_task),
                "success": int(successes[-1]),
                "started_task": int(bool(info.get("task_started", False))),
                "started_on_time": int(starts[-1]),
                "cumulative_success_rate": cumulative_success_rate,
                "rolling_success_rate": cumulative_success_rate,
                "cumulative_start_rate": cumulative_start_rate,
                "rolling_start_rate": rolling_start_rate,
            }
        )

    stats = {
        "mean_reward": float(np.mean(rewards)),
        "std_reward":  float(np.std(rewards)),
        "mean_length": float(np.mean(lengths)),
        "success_rate": float(np.mean(successes)),
        "start_rate": float(np.mean(starts)),
        "episodes": args.episodes,
    }

    print(f"\nEvaluation over {args.episodes} episodes:")
    print(f"  Success criterion: placement within {args.success_time_limit} steps")
    print(f"  Mean reward  : {stats['mean_reward']:.3f} ± {stats['std_reward']:.3f}")
    print(f"  Mean length  : {stats['mean_length']:.1f}")
    print(f"  Timed success rate : {stats['success_rate']:.3f}")
    print(
        f"  Start rate   : {stats['start_rate']:.3f} "
        f"(strict + {args.start_grace_steps} grace steps)"
    )

    os.makedirs(args.results_dir, exist_ok=True)
    metrics_path = os.path.join(args.results_dir, args.metrics_filename)
    save_evaluation_metrics(metrics_path, rows)
    print(f"  Evaluation metrics saved → {metrics_path}")

    success_plot_path = os.path.join(args.results_dir, args.success_plot_filename)
    reward_plot_path = os.path.join(args.results_dir, args.reward_plot_filename)
    try:
        generate_plots_from_metrics(
            metrics_path=metrics_path,
            success_out=success_plot_path,
            reward_out=reward_plot_path,
            rolling_window=max(10, args.episodes // 10),
            title_prefix="Evaluation",
        )
        print(f"  Success-rate plot saved → {success_plot_path}")
        print(f"  Reward plot saved → {reward_plot_path}")
    except ModuleNotFoundError as exc:
        if exc.name == "matplotlib":
            print("  Plot generation skipped: matplotlib not available in this interpreter.")
            print("  Run plots with: .venv/bin/python results.py")
        else:
            raise

    return stats


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
