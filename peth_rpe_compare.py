"""
Generate a direct normal-vs-Parkinson PETH comparison plot.

The figure overlays both models on fixed y-limits so magnitude differences are
visible at a glance for beginning/middle/end phases.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import types


def ensure_runtime() -> None:
    needed = ("numpy", "matplotlib", "torch")
    if all(importlib.util.find_spec(pkg) is not None for pkg in needed):
        return

    if os.environ.get("HB_PETH_COMPARE_REEXEC") == "1":
        return

    repo_root = os.path.dirname(os.path.abspath(__file__))
    venv_python = os.path.join(repo_root, ".venv", "bin", "python")
    if os.path.exists(venv_python):
        env = os.environ.copy()
        env["HB_PETH_COMPARE_REEXEC"] = "1"
        os.execve(venv_python, [venv_python, os.path.abspath(__file__), *sys.argv[1:]], env)


ensure_runtime()

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mplconfig"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np

import peth_rpe
import peth_rpe_parkinsons


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare normal and Parkinson PETH on shared y-scale.")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints")
    parser.add_argument("--final_checkpoint", type=str, default="checkpoints/a2c_rpe_final.pt")
    parser.add_argument("--training_metrics", type=str, default="results/training_metrics.csv")
    parser.add_argument("--probe_episodes", type=int, default=200)
    parser.add_argument("--window_pre", type=int, default=10)
    parser.add_argument("--window_post", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--grid_size", type=int, default=5)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--reward_threshold", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="results/rpe_peth_model_comparison.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pre = max(0, int(args.window_pre))
    post = max(1, int(args.window_post))

    phase_args = types.SimpleNamespace(
        checkpoints_dir=args.checkpoints_dir,
        final_checkpoint=args.final_checkpoint,
        training_metrics=args.training_metrics,
        out="-",
        counts_out="-",
        probe_episodes=args.probe_episodes,
        window_pre=pre,
        window_post=post,
        gamma=args.gamma,
        grid_size=args.grid_size,
        hidden_dim=args.hidden_dim,
        max_steps=args.max_steps,
        reward_threshold=args.reward_threshold,
        seed=args.seed,
    )

    total_episodes = peth_rpe._read_total_episodes(args.training_metrics)
    midpoint_target = None if total_episodes is None else int(total_episodes * 0.5)
    middle_ckpt = peth_rpe.choose_middle_checkpoint(args.checkpoints_dir, midpoint_target)
    end_ckpt = args.final_checkpoint if os.path.exists(args.final_checkpoint) else None

    phases = [
        ("beginning", None),
        ("middle", middle_ckpt),
        ("end", end_ckpt),
    ]

    rows = []
    for phase_name, ckpt_path in phases:
        if phase_name != "beginning" and ckpt_path is None:
            continue
        normal = peth_rpe.collect_phase_windows(phase_name, ckpt_path, phase_args, pre, post)
        parkinson = peth_rpe_parkinsons.collect_phase_windows(phase_name, ckpt_path, phase_args, pre, post)
        rows.append((phase_name, normal, parkinson))

    if not rows:
        raise RuntimeError("No valid phases found for comparison.")

    x = np.arange(-pre, post + 1)
    max_abs = 0.0
    for _, n, p in rows:
        for arr in (n["cue_mean"], n["reward_mean"], p["cue_mean"], p["reward_mean"]):
            with np.errstate(invalid="ignore"):
                cur = float(np.nanmax(np.abs(arr))) if np.any(~np.isnan(arr)) else 0.0
            max_abs = max(max_abs, cur)
    y_lim = max(0.02, 1.1 * max_abs)

    fig, axes = plt.subplots(len(rows), 2, figsize=(12, 3.4 * len(rows)), sharex=True, sharey=True)
    if len(rows) == 1:
        axes = np.array([axes])

    for r, (phase_name, n, p) in enumerate(rows):
        ax_cue = axes[r, 0]
        ax_rew = axes[r, 1]

        ax_cue.plot(x, n["cue_mean"], label=f"Normal (n={n['cue_events']})", linewidth=2.0, color="#1f77b4")
        ax_cue.plot(x, p["cue_mean"], label=f"Parkinson (n={p['cue_events']})", linewidth=2.0, color="#d62728")
        ax_cue.axvline(0, linestyle="--", linewidth=1.1, color="black")
        ax_cue.axhline(0, linestyle=":", linewidth=0.9, color="gray")
        ax_cue.set_ylim(-y_lim, y_lim)
        ax_cue.grid(alpha=0.25)
        ax_cue.set_title(f"{phase_name.capitalize()} cue-aligned")

        ax_rew.plot(x, n["reward_mean"], label=f"Normal (n={n['reward_events']})", linewidth=2.0, color="#1f77b4")
        ax_rew.plot(x, p["reward_mean"], label=f"Parkinson (n={p['reward_events']})", linewidth=2.0, color="#d62728")
        ax_rew.axvline(0, linestyle="--", linewidth=1.1, color="black")
        ax_rew.axhline(0, linestyle=":", linewidth=0.9, color="gray")
        ax_rew.set_ylim(-y_lim, y_lim)
        ax_rew.grid(alpha=0.25)
        ax_rew.set_title(f"{phase_name.capitalize()} reward-aligned")

        if r == len(rows) - 1:
            ax_cue.set_xlabel("Steps from event")
            ax_rew.set_xlabel("Steps from event")
        ax_cue.set_ylabel("TD/RPE")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=True)
    fig.suptitle("Normal vs Parkinson PETH comparison (shared y-scale)", fontsize=13)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(args.out, dpi=150)
    plt.close(fig)

    print(f"Saved comparison PETH -> {args.out}")


if __name__ == "__main__":
    main()
