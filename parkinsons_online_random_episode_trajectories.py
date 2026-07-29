"""
Run Parkinsons online evaluation episodes, then plot random sampled trajectories.

This mirrors the random-trajectory sampler workflow, but uses the online-updating
Parkinson agent during rollout.

Usage
-----
python parkinsons_online_random_episode_trajectories.py
python parkinsons_online_random_episode_trajectories.py --episodes 200 --episodes_to_plot 10
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import random
import sys
from typing import Any


def _ensure_runtime() -> None:
    required = ("numpy", "torch", "matplotlib")
    if all(importlib.util.find_spec(name) is not None for name in required):
        return

    if os.environ.get("HB_PARKINSONS_ONLINE_TRAJ_REEXEC") == "1":
        return

    repo_root = os.path.dirname(os.path.abspath(__file__))
    venv_python = os.path.join(repo_root, ".venv", "bin", "python")
    if os.path.exists(venv_python):
        env = os.environ.copy()
        env["HB_PARKINSONS_ONLINE_TRAJ_REEXEC"] = "1"
        os.execve(venv_python, [venv_python, os.path.abspath(__file__), *sys.argv[1:]], env)


_ensure_runtime()

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mplconfig"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import numpy as np
import torch

from parkinsons_a2c_rpe_model import A2CAgent
from pick_and_place_env import PickAndPlaceEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Parkinsons online evaluation episodes and plot random sampled trajectories."
    )
    parser.add_argument("--checkpoint", type=str, default="checkpoints/a2c_rpe_final.pt")
    parser.add_argument("--grid_size", type=int, default=5)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument(
        "--n_steps",
        type=int,
        default=10,
        help="Number of environment steps per online update.",
    )
    parser.add_argument(
        "--episodes_to_plot",
        type=int,
        default=10,
        help="Number of random episodes to plot from the online run.",
    )
    parser.add_argument(
        "--success_time_limit",
        type=int,
        default=75,
        help="Timed-success threshold used for the sampled episode summaries.",
    )
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument(
        "--plot_filename",
        type=str,
        default="parkinsons_online_random_episode_trajectories.png",
    )
    parser.add_argument(
        "--summary_filename",
        type=str,
        default="parkinsons_online_random_episode_trajectories.csv",
    )
    parser.add_argument(
        "--transitions_filename",
        type=str,
        default="parkinsons_online_random_episode_trajectories_transitions.csv",
    )
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--save_all_episodes",
        action="store_true",
        help="Write summaries for every online episode, not just the sampled subset.",
    )
    return parser.parse_args()


def _resolve_checkpoint_path(path: str) -> str:
    if os.path.exists(path):
        return path

    candidate = os.path.join("checkpoints", os.path.basename(path))
    if os.path.exists(candidate):
        return candidate

    raise FileNotFoundError(path)


def _sample_episode_numbers(episode_numbers: list[int], n: int, rng: random.Random) -> list[int]:
    if len(episode_numbers) < n:
        raise ValueError(f"Requested {n} episodes but only {len(episode_numbers)} available.")
    return sorted(rng.sample(episode_numbers, n))


def _plot_trajectory_grid(
    trajectories: list[dict[str, Any]],
    grid_size: int,
    title: str,
    out_path: str,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib import colors as mcolors

    def _add_colored_path(ax, path: list[tuple[int, int]]) -> None:
        if len(path) < 2:
            return

        ys = np.array([p[0] for p in path], dtype=np.float32)
        xs = np.array([p[1] for p in path], dtype=np.float32)
        points = np.column_stack((xs, ys))
        segments = np.stack([points[:-1], points[1:]], axis=1)
        values = np.linspace(0.0, 1.0, len(segments), dtype=np.float32)
        norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
        lc = LineCollection(
            segments,
            cmap="viridis",
            norm=norm,
            linewidth=2.0,
            alpha=0.95,
        )
        lc.set_array(values)
        ax.add_collection(lc)

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

        _add_colored_path(ax, path)
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
            f"TS={item['timed_success']} | R={item['reward']:.1f}",
            fontsize=9,
        )

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)

    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0.0, 1.0))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes_flat.tolist(), shrink=0.8, pad=0.01)
    cbar.set_label("Earlier to later steps", rotation=90)

    fig.suptitle(title, fontsize=14)
    fig.subplots_adjust(top=0.88, wspace=0.18, hspace=0.28)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _write_rows(path: str, rows: list[dict[str, Any]]) -> None:
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


def _write_transitions(path: str, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "episode",
        "step",
        "state",
        "action",
        "action_prob",
        "reward",
        "next_state",
        "done",
        "terminated",
        "truncated",
        "cue_active",
        "task_started",
        "object_placed",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    checkpoint_path = _resolve_checkpoint_path(args.checkpoint)

    env = PickAndPlaceEnv(grid_size=args.grid_size, max_steps=args.max_steps)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = A2CAgent(state_dim=state_dim, action_dim=action_dim, hidden_dim=args.hidden_dim)
    agent.load(checkpoint_path)

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Online update interval: {args.n_steps} steps")
    print(f"Episodes: {args.episodes}")

    rng = random.Random(args.sample_seed)
    episode_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    episode_trajectories: list[dict[str, Any]] = []

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        trajectory = [tuple(env.agent_pos.tolist())]
        ep_reward = 0.0
        ep_length = 0
        terminated = False
        truncated = False
        last_info: dict[str, Any] = {}

        while not (terminated or truncated):
            batch_s, batch_a, batch_r, batch_ns, batch_d, batch_old_p = [], [], [], [], [], []

            for _ in range(args.n_steps):
                if args.render:
                    env.render()

                action, action_probs = agent.select_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)

                transition_rows.append(
                    {
                        "episode": int(ep + 1),
                        "step": int(ep_length + 1),
                        "state": json.dumps(np.asarray(obs, dtype=np.float32).tolist()),
                        "action": int(action),
                        "action_prob": float(action_probs[action].item()),
                        "reward": float(reward),
                        "next_state": json.dumps(np.asarray(next_obs, dtype=np.float32).tolist()),
                        "done": int(terminated or truncated),
                        "terminated": int(terminated),
                        "truncated": int(truncated),
                        "cue_active": int(bool(info.get("cue_active", False))),
                        "task_started": int(bool(info.get("task_started", False))),
                        "object_placed": int(bool(info.get("object_placed", False))),
                    }
                )

                batch_s.append(obs)
                batch_a.append(action)
                batch_r.append(float(reward))
                batch_ns.append(next_obs)
                batch_d.append(float(terminated or truncated))
                batch_old_p.append(float(action_probs[action].item()))

                ep_reward += float(reward)
                ep_length += 1
                obs = next_obs
                trajectory.append(tuple(env.agent_pos.tolist()))
                last_info = info

                if terminated or truncated:
                    break

            if batch_s:
                agent.update(batch_s, batch_a, batch_r, batch_ns, batch_d, batch_old_p)

        completed = bool(last_info.get("object_placed", False))
        timed_success = int(completed and ep_length <= args.success_time_limit)

        row = {
            "episode": ep + 1,
            "seed": args.seed + ep,
            "episode_length": int(ep_length),
            "reward": float(ep_reward),
            "completed_task": int(completed),
            "timed_success": int(timed_success),
        }
        episode_rows.append(row)
        episode_trajectories.append(
            {
                **row,
                "trajectory": trajectory,
                "object_pos": tuple(env.object_pos.tolist()),
                "target_pos": tuple(env.target_pos.tolist()),
            }
        )

        # Advance the degeneration schedule (motivation-neuron pruning plus the
        # coupled RPE impairment decay), continuing from the checkpoint state.
        if hasattr(agent, "on_episode_end"):
            agent.on_episode_end()

    sampled_episode_numbers = _sample_episode_numbers(
        [row["episode"] for row in episode_rows],
        args.episodes_to_plot,
        rng,
    )
    sampled_rows = [row for row in episode_trajectories if row["episode"] in sampled_episode_numbers]

    plot_path = os.path.join(args.results_dir, args.plot_filename)
    summary_path = os.path.join(args.results_dir, args.summary_filename)
    transitions_path = os.path.join(args.results_dir, args.transitions_filename)

    _plot_trajectory_grid(
        sampled_rows,
        grid_size=args.grid_size,
        title=(
            f"Parkinsons Online: {args.episodes_to_plot} Random Episodes "
            f"(sample_seed={args.sample_seed}, n_steps={args.n_steps})"
        ),
        out_path=plot_path,
    )
    _write_rows(summary_path, sampled_rows)
    _write_transitions(transitions_path, transition_rows)

    if args.save_all_episodes:
        all_summary_path = os.path.join(args.results_dir, "parkinsons_online_all_episode_trajectories.csv")
        _write_rows(all_summary_path, episode_trajectories)
        print(f"Saved all-episode summary -> {all_summary_path}")

    print(f"Sampled episodes: {sampled_episode_numbers}")
    print(f"Saved trajectory plot -> {plot_path}")
    print(f"Saved sampled summary -> {summary_path}")
    print(f"Saved transition log -> {transitions_path}")


if __name__ == "__main__":
    main()