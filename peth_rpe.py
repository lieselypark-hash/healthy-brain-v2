"""
Generate a PETH-adjacent diagram for TD/RPE around cue and reward events.

The figure overlays three training phases:
- beginning: randomly initialized model (before training)
- middle: closest checkpoint to halfway through training
- end: final checkpoint

Usage
-----
python peth_rpe.py
python peth_rpe.py --probe_episodes 300 --window_pre 10 --window_post 20
"""

from __future__ import annotations

import argparse
import csv
import glob
import importlib.util
import os
import re
import sys
from typing import Optional


def ensure_runtime() -> None:
    """Re-launch with workspace venv if plotting or torch deps are missing."""
    needed = ("numpy", "torch", "matplotlib")
    if all(importlib.util.find_spec(pkg) is not None for pkg in needed):
        return

    if os.environ.get("HB_PETH_REEXEC") == "1":
        return

    repo_root = os.path.dirname(os.path.abspath(__file__))
    venv_python = os.path.join(repo_root, ".venv", "bin", "python")
    if os.path.exists(venv_python):
        env = os.environ.copy()
        env["HB_PETH_REEXEC"] = "1"
        os.execve(venv_python, [venv_python, os.path.abspath(__file__), *sys.argv[1:]], env)


ensure_runtime()

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mplconfig"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import numpy as np
import torch
import matplotlib.pyplot as plt

from a2c_rpe_model import A2CAgent
from pick_and_place_env import PickAndPlaceEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot RPE traces aligned to cue onset and reward events for beginning/middle/end training phases.",
    )
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints")
    parser.add_argument("--final_checkpoint", type=str, default="checkpoints/a2c_rpe_final.pt")
    parser.add_argument("--training_metrics", type=str, default="results/training_metrics.csv")
    parser.add_argument("--out", type=str, default="results/rpe_peth_begin_mid_end.png")
    parser.add_argument("--counts_out", type=str, default="results/rpe_peth_event_counts.csv")
    parser.add_argument("--probe_episodes", type=int, default=200,
                        help="Episodes sampled per phase to estimate event-aligned RPE.")
    parser.add_argument("--window_pre", type=int, default=10,
                        help="Steps before event in the peri-event window.")
    parser.add_argument("--window_post", type=int, default=20,
                        help="Steps after event in the peri-event window.")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor used to compute TD/RPE during probing.")
    parser.add_argument("--grid_size", type=int, default=5)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--reward_threshold", type=float, default=2.0,
                        help="Reward-event threshold. Default 2.0 captures pick/place rewards.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--evaluation_reward_policy",
        action="store_true",
        help="Match evaluation reward policy (no shaping; only pick/place rewards).",
    )
    return parser.parse_args()


def _read_total_episodes(training_metrics_path: str) -> Optional[int]:
    if not os.path.exists(training_metrics_path):
        return None

    last_episode = None
    with open(training_metrics_path, "r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            try:
                last_episode = int(row.get("episode", ""))
            except ValueError:
                continue
    return last_episode


def _extract_episode_from_ckpt(path: str) -> Optional[int]:
    m = re.search(r"a2c_rpe_ep(\d+)\.pt$", os.path.basename(path))
    if not m:
        return None
    return int(m.group(1))


def choose_middle_checkpoint(checkpoints_dir: str, target_episode: Optional[int]) -> Optional[str]:
    candidates = glob.glob(os.path.join(checkpoints_dir, "a2c_rpe_ep*.pt"))
    ep_paths = []
    for path in candidates:
        ep = _extract_episode_from_ckpt(path)
        if ep is not None:
            ep_paths.append((ep, path))
    if not ep_paths:
        return None

    if target_episode is None:
        max_ep = max(ep for ep, _ in ep_paths)
        target_episode = max_ep // 2

    ep_paths.sort(key=lambda x: abs(x[0] - target_episode))
    return ep_paths[0][1]


def compute_td_rpe(agent: A2CAgent, state: np.ndarray, reward: float,
                   next_state: np.ndarray, done: bool, gamma: float) -> float:
    with torch.no_grad():
        s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        ns = torch.as_tensor(next_state, dtype=torch.float32).unsqueeze(0)
        _, v = agent.network(s)
        _, nv = agent.network(ns)
        v_scalar = float(v.squeeze().item())
        nv_scalar = 0.0 if done else float(nv.squeeze().item())
    return float(reward + gamma * nv_scalar - v_scalar)


def compute_td_components(
    agent: A2CAgent,
    state: np.ndarray,
    reward: float,
    next_state: np.ndarray,
    done: bool,
    gamma: float,
) -> tuple[float, float, float, float]:
    with torch.no_grad():
        s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        ns = torch.as_tensor(next_state, dtype=torch.float32).unsqueeze(0)
        _, v = agent.network(s)
        _, nv = agent.network(ns)
        value_t = float(v.squeeze().item())
        value_t1 = 0.0 if done else float(nv.squeeze().item())
        reward_value = float(reward)
    rpe = float(reward_value + gamma * value_t1 - value_t)
    return reward_value, value_t1, value_t, rpe


def extract_window(trace: list[float], center_idx: int, pre: int, post: int) -> np.ndarray:
    window = np.full(pre + post + 1, np.nan, dtype=np.float32)
    for offset in range(-pre, post + 1):
        src = center_idx + offset
        dst = offset + pre
        if 0 <= src < len(trace):
            window[dst] = trace[src]
    return window


def aggregate_windows(windows: list[np.ndarray], width: int) -> tuple[np.ndarray, np.ndarray, int]:
    if not windows:
        nan_arr = np.full(width, np.nan, dtype=np.float32)
        return nan_arr, nan_arr, 0

    mat = np.vstack(windows)
    valid_mask = ~np.isnan(mat)
    valid_n = np.sum(valid_mask, axis=0).astype(np.float32)

    safe_mat = np.where(valid_mask, mat, 0.0)
    sum_vals = np.sum(safe_mat, axis=0)

    mean = np.full(width, np.nan, dtype=np.float32)
    has_data = valid_n > 0
    mean[has_data] = (sum_vals[has_data] / valid_n[has_data]).astype(np.float32)

    centered = np.where(valid_mask, safe_mat - mean[None, :], 0.0)
    sq = np.sum(centered * centered, axis=0)

    std = np.full(width, np.nan, dtype=np.float32)
    std[has_data] = np.sqrt((sq[has_data] / valid_n[has_data]).astype(np.float32))

    sem = np.full(width, np.nan, dtype=np.float32)
    sem[has_data] = std[has_data] / np.sqrt(valid_n[has_data])
    return mean.astype(np.float32), sem.astype(np.float32), mat.shape[0]


def build_agent_for_env(env: PickAndPlaceEnv, hidden_dim: int, gamma: float) -> A2CAgent:
    return A2CAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        hidden_dim=hidden_dim,
        gamma=gamma,
    )


def collect_phase_windows(
    phase_name: str,
    ckpt_path: Optional[str],
    args: argparse.Namespace,
    pre: int,
    post: int,
) -> dict:
    env_kwargs = {
        "grid_size": args.grid_size,
        "max_steps": args.max_steps,
        "seed": args.seed,
    }
    if getattr(args, "evaluation_reward_policy", False):
        env_kwargs.update({"shaping_start": 0.0, "shaping_end": 0.0})
    env = PickAndPlaceEnv(**env_kwargs)
    if getattr(args, "evaluation_reward_policy", False):
        env.reward_step = 0.0
        env.reward_invalid = 0.0
        env.reward_start = 0.0
        env.shaping_scale = 0.0
    agent = build_agent_for_env(env, args.hidden_dim, args.gamma)

    if ckpt_path is not None:
        agent.load(ckpt_path)

    cue_windows: list[np.ndarray] = []
    reward_windows: list[np.ndarray] = []
    reward_component_windows: list[np.ndarray] = []
    next_value_windows: list[np.ndarray] = []
    value_windows: list[np.ndarray] = []
    rpe_windows: list[np.ndarray] = []

    for ep in range(args.probe_episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        prev_cue_active = False
        rpe_trace: list[float] = []
        reward_trace: list[float] = []
        next_value_trace: list[float] = []
        value_trace: list[float] = []
        cue_events: list[int] = []
        reward_events: list[int] = []

        while not done:
            action, _ = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            reward_value, value_t1, value_t, rpe = compute_td_components(
                agent,
                obs,
                reward,
                next_obs,
                done,
                args.gamma,
            )
            reward_trace.append(reward_value)
            next_value_trace.append(value_t1)
            value_trace.append(value_t)
            rpe_trace.append(rpe)

            cue_active = bool(info.get("cue_active", False))
            if cue_active and not prev_cue_active:
                cue_events.append(len(rpe_trace) - 1)
            prev_cue_active = cue_active

            if reward >= args.reward_threshold:
                reward_events.append(len(rpe_trace) - 1)

            obs = next_obs

        for idx in cue_events:
            cue_windows.append(extract_window(rpe_trace, idx, pre, post))
        for idx in reward_events:
            reward_windows.append(extract_window(rpe_trace, idx, pre, post))
            reward_component_windows.append(extract_window(reward_trace, idx, pre, post))
            next_value_windows.append(extract_window(next_value_trace, idx, pre, post))
            value_windows.append(extract_window(value_trace, idx, pre, post))
            rpe_windows.append(extract_window(rpe_trace, idx, pre, post))

        if hasattr(agent, "on_episode_end"):
            agent.on_episode_end()

    width = pre + post + 1
    cue_mean, cue_sem, cue_n = aggregate_windows(cue_windows, width)
    rew_mean, rew_sem, rew_n = aggregate_windows(reward_windows, width)
    reward_component_mean, reward_component_sem, _ = aggregate_windows(reward_component_windows, width)
    next_value_mean, next_value_sem, _ = aggregate_windows(next_value_windows, width)
    value_mean, value_sem, _ = aggregate_windows(value_windows, width)
    rpe_component_mean, rpe_component_sem, _ = aggregate_windows(rpe_windows, width)

    return {
        "phase": phase_name,
        "checkpoint": ckpt_path or "<random_init>",
        "cue_mean": cue_mean,
        "cue_sem": cue_sem,
        "cue_events": cue_n,
        "reward_mean": rew_mean,
        "reward_sem": rew_sem,
        "reward_events": rew_n,
        "reward_component_mean": reward_component_mean,
        "reward_component_sem": reward_component_sem,
        "next_value_mean": next_value_mean,
        "next_value_sem": next_value_sem,
        "value_mean": value_mean,
        "value_sem": value_sem,
        "rpe_component_mean": rpe_component_mean,
        "rpe_component_sem": rpe_component_sem,
    }


def save_counts(path: str, phase_data: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=["phase", "checkpoint", "cue_events", "reward_events"],
        )
        writer.writeheader()
        for row in phase_data:
            writer.writerow(
                {
                    "phase": row["phase"],
                    "checkpoint": row["checkpoint"],
                    "cue_events": row["cue_events"],
                    "reward_events": row["reward_events"],
                }
            )


def plot_peth(phase_data: list[dict], out_path: str, pre: int, post: int) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    x = np.arange(-pre, post + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    color_map = {
        "beginning": "#1f77b4",
        "middle": "#ff7f0e",
        "end": "#2ca02c",
    }

    for row in phase_data:
        label_cue = f"{row['phase']} (n={row['cue_events']})"
        label_rew = f"{row['phase']} (n={row['reward_events']})"
        color = color_map.get(row["phase"], None)

        axes[0].plot(x, row["cue_mean"], label=label_cue, linewidth=2, color=color)
        axes[0].fill_between(
            x,
            row["cue_mean"] - row["cue_sem"],
            row["cue_mean"] + row["cue_sem"],
            alpha=0.18,
            color=color,
        )

        axes[1].plot(x, row["reward_mean"], label=label_rew, linewidth=2, color=color)
        axes[1].fill_between(
            x,
            row["reward_mean"] - row["reward_sem"],
            row["reward_mean"] + row["reward_sem"],
            alpha=0.18,
            color=color,
        )

    for ax, title in zip(
        axes,
        ["Cue-aligned TD/RPE", "Reward-aligned TD/RPE"],
    ):
        ax.axvline(0, linestyle="--", linewidth=1.3, color="black")
        ax.axhline(0, linestyle=":", linewidth=1.0, color="gray")
        ax.set_title(title)
        ax.set_xlabel("Steps from event")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("TD/RPE (delta)")
    axes[0].legend(loc="best", frameon=True)
    axes[1].legend(loc="best", frameon=True)

    fig.suptitle("PETH-adjacent RPE profiles at beginning/middle/end of training", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_rpe_components(phase_data: list[dict], out_path: str, pre: int, post: int) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    x = np.arange(-pre, post + 1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.6), sharex=True)
    component_specs = [
        ("reward_component_mean", "reward_component_sem", "Reward"),
        ("next_value_mean", "next_value_sem", "Value at state t+1"),
        ("value_mean", "value_sem", "Value at state t"),
        ("rpe_component_mean", "rpe_component_sem", "TD/RPE"),
    ]
    color_map = {
        "beginning": "#1f77b4",
        "middle": "#ff7f0e",
        "end": "#2ca02c",
    }

    for ax, (mean_key, sem_key, title) in zip(axes.flat, component_specs):
        for row in phase_data:
            color = color_map.get(row["phase"], None)
            mean = row[mean_key]
            sem = row[sem_key]
            label = f"{row['phase']} (n={row['reward_events']})"
            ax.plot(x, mean, label=label, linewidth=2.0, color=color)
            ax.fill_between(
                x,
                mean - sem,
                mean + sem,
                alpha=0.16,
                color=color,
            )

        ax.axvline(0, linestyle="--", linewidth=1.2, color="black")
        ax.axhline(0, linestyle=":", linewidth=1.0, color="gray")
        ax.set_title(title)
        ax.set_xlabel("Steps from reward event")
        ax.grid(alpha=0.25)

    axes[0, 0].set_ylabel("Signal magnitude")
    axes[1, 0].set_ylabel("Signal magnitude")
    axes[0, 0].legend(loc="best", frameon=True)

    fig.suptitle("RPE components at beginning/middle/end of training", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    pre = max(0, int(args.window_pre))
    post = max(1, int(args.window_post))

    total_episodes = _read_total_episodes(args.training_metrics)
    midpoint_target = None if total_episodes is None else int(total_episodes * 0.5)
    middle_ckpt = choose_middle_checkpoint(args.checkpoints_dir, midpoint_target)

    end_ckpt = args.final_checkpoint if os.path.exists(args.final_checkpoint) else None
    if end_ckpt is None:
        all_ckpts = glob.glob(os.path.join(args.checkpoints_dir, "a2c_rpe_ep*.pt"))
        ep_ckpts = [( _extract_episode_from_ckpt(p), p) for p in all_ckpts]
        ep_ckpts = [(ep, p) for ep, p in ep_ckpts if ep is not None]
        if ep_ckpts:
            end_ckpt = sorted(ep_ckpts, key=lambda x: x[0])[-1][1]

    phases = [
        ("beginning", None),
        ("middle", middle_ckpt),
        ("end", end_ckpt),
    ]

    phase_data = []
    for phase_name, ckpt_path in phases:
        if phase_name != "beginning" and ckpt_path is None:
            continue
        phase_data.append(collect_phase_windows(phase_name, ckpt_path, args, pre, post))

    if not phase_data:
        raise RuntimeError("No valid phases found. Train a model/checkpoints first.")

    plot_peth(phase_data, args.out, pre, post)
    components_out = os.path.join(os.path.dirname(args.out), "rpe_components_begin_mid_end.png")
    plot_rpe_components(phase_data, components_out, pre, post)
    save_counts(args.counts_out, phase_data)

    print(f"Saved PETH-adjacent figure -> {args.out}")
    print(f"Saved RPE component figure  -> {components_out}")
    print(f"Saved event counts       -> {args.counts_out}")
    for row in phase_data:
        print(
            f"  {row['phase']:<9} cue_events={row['cue_events']:<6d} "
            f"reward_events={row['reward_events']:<6d} checkpoint={row['checkpoint']}"
        )


if __name__ == "__main__":
    main()
