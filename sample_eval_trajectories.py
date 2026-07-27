"""
Sample random evaluation episodes and plot their replayed trajectories.

Usage
-----
python sample_eval_trajectories.py
python sample_eval_trajectories.py --episodes_per_group 10 --sample_seed 42
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import random
import sys
from typing import Any


def _ensure_runtime() -> None:
    required = ("numpy", "torch", "matplotlib")
    if all(importlib.util.find_spec(name) is not None for name in required):
        return

    if os.environ.get("HB_TRAJ_REEXEC") == "1":
        return

    repo_root = os.path.dirname(os.path.abspath(__file__))
    venv_python = os.path.join(repo_root, ".venv", "bin", "python")
    if os.path.exists(venv_python):
        env = os.environ.copy()
        env["HB_TRAJ_REEXEC"] = "1"
        os.execve(venv_python, [venv_python, os.path.abspath(__file__), *sys.argv[1:]], env)


_ensure_runtime()

import numpy as np
import torch

from a2c_rpe_model import A2CAgent as NormalA2CAgent
from parkinsons_a2c_rpe_model import A2CAgent as ParkinsonsA2CAgent
from pick_and_place_env import PickAndPlaceEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select random episodes from evaluation CSVs and plot replayed trajectories."
    )
    parser.add_argument(
        "--normal_metrics",
        type=str,
        default="results/evaluation_normal_metrics.csv",
    )
    parser.add_argument(
        "--parkinsons_metrics",
        type=str,
        default="results/evaluation_parkinsons_metrics.csv",
    )
    parser.add_argument(
        "--normal_checkpoint",
        type=str,
        default="checkpoints/a2c_rpe_final.pt",
    )
    parser.add_argument(
        "--parkinsons_checkpoint",
        type=str,
        default="checkpoints/a2c_rpe_final.pt",
    )
    parser.add_argument("--grid_size", type=int, default=5)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument(
        "--eval_seed",
        type=int,
        default=0,
        help="Base seed used in evaluation scripts (episode seed is eval_seed + episode_index).",
    )
    parser.add_argument(
        "--episodes_per_group",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--sample_seed",
        type=int,
        default=42,
        help="Seed used to randomly select episode IDs from each evaluation CSV.",
    )
    parser.add_argument(
        "--success_time_limit",
        type=int,
        default=100,
        help="Timed-success threshold used for reporting in figure titles.",
    )
    parser.add_argument("--out_dir", type=str, default="results")
    return parser.parse_args()


def _read_episode_rows(path: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metrics file not found: {path}")
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            row["episode"] = int(row["episode"])
            row["episode_length"] = int(float(row.get("episode_length", "0") or 0))
            row["completed_task"] = int(float(row.get("completed_task", row.get("success", "0")) or 0))
            row["success"] = int(float(row.get("success", "0") or 0))
            rows.append(row)
    if not rows:
        raise ValueError(f"Metrics file is empty: {path}")
    return rows


def _build_agent(kind: str, state_dim: int, action_dim: int, hidden_dim: int, checkpoint: str):
    if kind == "normal":
        agent = NormalA2CAgent(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim)
    elif kind == "parkinsons":
        agent = ParkinsonsA2CAgent(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim)
    else:
        raise ValueError(f"Unknown agent kind: {kind}")
    agent.load(checkpoint)
    return agent


def _replay_episode(
    agent,
    env: PickAndPlaceEnv,
    episode_number: int,
    eval_seed: int,
    success_time_limit: int,
) -> dict[str, Any]:
    episode_index = episode_number - 1
    replay_seed = eval_seed + episode_index

    np.random.seed(replay_seed)
    torch.manual_seed(replay_seed)

    obs, _ = env.reset(seed=replay_seed)
    trajectory = [tuple(env.agent_pos.tolist())]

    ep_reward = 0.0
    ep_length = 0
    terminated = False
    truncated = False
    last_info: dict[str, Any] = {}

    while not (terminated or truncated):
        action, _ = agent.select_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward
        ep_length += 1
        trajectory.append(tuple(env.agent_pos.tolist()))
        last_info = info

    completed = bool(last_info.get("object_placed", False))
    timed_success = completed and ep_length <= success_time_limit

    return {
        "episode": episode_number,
        "seed": replay_seed,
        "trajectory": trajectory,
        "reward": float(ep_reward),
        "episode_length": int(ep_length),
        "completed_task": int(completed),
        "timed_success": int(timed_success),
        "start_step": trajectory[0],
        "end_step": trajectory[-1],
        "object_pos": tuple(env.object_pos.tolist()),
        "target_pos": tuple(env.target_pos.tolist()),
    }


def _plot_trajectory_grid(
    trajectories: list[dict[str, Any]],
    grid_size: int,
    title: str,
    out_path: str,
) -> None:
    import matplotlib.pyplot as plt

    cols = 5
    rows = 2
    fig, axes = plt.subplots(rows, cols, figsize=(22, 9), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for idx, ax in enumerate(axes_flat):
        if idx >= len(trajectories):
            ax.axis("off")
            continue

        item = trajectories[idx]
        path = item["trajectory"]
        ys = [p[0] for p in path]
        xs = [p[1] for p in path]

        ax.plot(xs, ys, "-o", markersize=2.2, linewidth=1.2, alpha=0.9)
        ax.scatter(xs[0], ys[0], c="green", s=40, marker="o", label="Start")
        ax.scatter(xs[-1], ys[-1], c="red", s=40, marker="x", label="End")

        obj_y, obj_x = item["object_pos"]
        tgt_y, tgt_x = item["target_pos"]
        ax.scatter(obj_x, obj_y, c="orange", s=70, marker="s", label="Object")
        ax.scatter(tgt_x, tgt_y, c="purple", s=80, marker="*", label="Target")

        ax.set_xlim(-0.5, grid_size - 0.5)
        ax.set_ylim(grid_size - 0.5, -0.5)
        ax.set_xticks(range(grid_size))
        ax.set_yticks(range(grid_size))
        ax.grid(alpha=0.25)

        ax.set_title(
            f"Ep {item['episode']} | L={item['episode_length']} | "
            f"C={item['completed_task']} | TS={item['timed_success']}",
            fontsize=9,
        )

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _write_selected_rows(path: str, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "episode",
        "seed",
        "episode_length",
        "reward",
        "completed_task",
        "timed_success",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})


def _sample_episode_numbers(rows: list[dict[str, Any]], n: int, rng: random.Random) -> list[int]:
    episode_numbers = [int(r["episode"]) for r in rows]
    if len(episode_numbers) < n:
        raise ValueError(
            f"Requested {n} episodes but only {len(episode_numbers)} available."
        )
    return sorted(rng.sample(episode_numbers, n))


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    normal_rows = _read_episode_rows(args.normal_metrics)
    pd_rows = _read_episode_rows(args.parkinsons_metrics)

    rng = random.Random(args.sample_seed)
    normal_eps = _sample_episode_numbers(normal_rows, args.episodes_per_group, rng)
    pd_eps = _sample_episode_numbers(pd_rows, args.episodes_per_group, rng)

    env_normal = PickAndPlaceEnv(grid_size=args.grid_size, max_steps=args.max_steps)
    env_pd = PickAndPlaceEnv(grid_size=args.grid_size, max_steps=args.max_steps)

    state_dim = env_normal.observation_space.shape[0]
    action_dim = env_normal.action_space.n

    normal_agent = _build_agent(
        kind="normal",
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        checkpoint=args.normal_checkpoint,
    )
    pd_agent = _build_agent(
        kind="parkinsons",
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        checkpoint=args.parkinsons_checkpoint,
    )

    normal_trajs = [
        _replay_episode(
            normal_agent,
            env_normal,
            episode_number=ep,
            eval_seed=args.eval_seed,
            success_time_limit=args.success_time_limit,
        )
        for ep in normal_eps
    ]

    pd_trajs = [
        _replay_episode(
            pd_agent,
            env_pd,
            episode_number=ep,
            eval_seed=args.eval_seed,
            success_time_limit=args.success_time_limit,
        )
        for ep in pd_eps
    ]

    normal_plot = os.path.join(args.out_dir, "normal_random_episode_trajectories.png")
    pd_plot = os.path.join(args.out_dir, "parkinsons_random_episode_trajectories.png")

    _plot_trajectory_grid(
        normal_trajs,
        grid_size=args.grid_size,
        title=(
            f"Normal Agent: {args.episodes_per_group} Random Evaluation Episodes "
            f"(sample_seed={args.sample_seed}, timed_limit={args.success_time_limit})"
        ),
        out_path=normal_plot,
    )
    _plot_trajectory_grid(
        pd_trajs,
        grid_size=args.grid_size,
        title=(
            f"Parkinson Agent: {args.episodes_per_group} Random Evaluation Episodes "
            f"(sample_seed={args.sample_seed}, timed_limit={args.success_time_limit})"
        ),
        out_path=pd_plot,
    )

    normal_csv = os.path.join(args.out_dir, "normal_random_episode_trajectories.csv")
    pd_csv = os.path.join(args.out_dir, "parkinsons_random_episode_trajectories.csv")
    _write_selected_rows(normal_csv, normal_trajs)
    _write_selected_rows(pd_csv, pd_trajs)

    print(f"Normal episodes sampled: {normal_eps}")
    print(f"Parkinson episodes sampled: {pd_eps}")
    print(f"Saved normal trajectory plot -> {normal_plot}")
    print(f"Saved Parkinson trajectory plot -> {pd_plot}")
    print(f"Saved normal trajectory summary -> {normal_csv}")
    print(f"Saved Parkinson trajectory summary -> {pd_csv}")


if __name__ == "__main__":
    main()
