"""
Generate average evaluation PETH comparison for normal vs Parkinson models.

This script computes event-aligned average TD/RPE traces over evaluation-style
trials (no shaping, reward only from pick/place) and overlays both models.
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

    if os.environ.get("HB_PETH_EVAL_COMPARE_REEXEC") == "1":
        return

    repo_root = os.path.dirname(os.path.abspath(__file__))
    venv_python = os.path.join(repo_root, ".venv", "bin", "python")
    if os.path.exists(venv_python):
        env = os.environ.copy()
        env["HB_PETH_EVAL_COMPARE_REEXEC"] = "1"
        os.execve(venv_python, [venv_python, os.path.abspath(__file__), *sys.argv[1:]], env)


ensure_runtime()

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mplconfig"),
)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np

import peth_rpe
import peth_rpe_parkinsons


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot average evaluation PETH for normal vs Parkinson models.",
    )
    parser.add_argument("--checkpoint", type=str, default="checkpoints/a2c_rpe_final.pt")
    parser.add_argument("--episodes", type=int, default=500,
                        help="Number of evaluation episodes to average over.")
    parser.add_argument("--window_pre", type=int, default=10)
    parser.add_argument("--window_post", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--grid_size", type=int, default=5)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--reward_threshold", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        type=str,
        default="results/rpe_peth_evaluation_avg_normal_vs_parkinsons.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pre = max(0, int(args.window_pre))
    post = max(1, int(args.window_post))

    phase_args = types.SimpleNamespace(
        checkpoints_dir="checkpoints",
        final_checkpoint=args.checkpoint,
        training_metrics="results/training_metrics.csv",
        out="-",
        counts_out="-",
        probe_episodes=args.episodes,
        window_pre=pre,
        window_post=post,
        gamma=args.gamma,
        grid_size=args.grid_size,
        hidden_dim=args.hidden_dim,
        max_steps=args.max_steps,
        reward_threshold=args.reward_threshold,
        seed=args.seed,
        evaluation_reward_policy=True,
    )

    normal = peth_rpe.collect_phase_windows(
        phase_name="evaluation",
        ckpt_path=args.checkpoint,
        args=phase_args,
        pre=pre,
        post=post,
    )
    parkinson = peth_rpe_parkinsons.collect_phase_windows(
        phase_name="evaluation",
        ckpt_path=args.checkpoint,
        args=phase_args,
        pre=pre,
        post=post,
    )

    x = np.arange(-pre, post + 1)
    max_abs = 0.0
    for arr in (
        normal["cue_mean"],
        normal["reward_mean"],
        parkinson["cue_mean"],
        parkinson["reward_mean"],
    ):
        with np.errstate(invalid="ignore"):
            cur = float(np.nanmax(np.abs(arr))) if np.any(~np.isnan(arr)) else 0.0
        max_abs = max(max_abs, cur)
    y_lim = max(0.02, 1.1 * max_abs)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)

    axes[0].plot(
        x,
        normal["cue_mean"],
        label=f"Normal (n={normal['cue_events']})",
        linewidth=2.0,
        color="#1f77b4",
    )
    axes[0].plot(
        x,
        parkinson["cue_mean"],
        label=f"Parkinson (n={parkinson['cue_events']})",
        linewidth=2.0,
        color="#d62728",
    )
    axes[0].axvline(0, linestyle="--", linewidth=1.1, color="black")
    axes[0].axhline(0, linestyle=":", linewidth=0.9, color="gray")
    axes[0].set_ylim(-y_lim, y_lim)
    axes[0].grid(alpha=0.25)
    axes[0].set_title("Cue-aligned evaluation average")
    axes[0].set_xlabel("Steps from event")
    axes[0].set_ylabel("TD/RPE")

    axes[1].plot(
        x,
        normal["reward_mean"],
        label=f"Normal (n={normal['reward_events']})",
        linewidth=2.0,
        color="#1f77b4",
    )
    axes[1].plot(
        x,
        parkinson["reward_mean"],
        label=f"Parkinson (n={parkinson['reward_events']})",
        linewidth=2.0,
        color="#d62728",
    )
    axes[1].axvline(0, linestyle="--", linewidth=1.1, color="black")
    axes[1].axhline(0, linestyle=":", linewidth=0.9, color="gray")
    axes[1].set_ylim(-y_lim, y_lim)
    axes[1].grid(alpha=0.25)
    axes[1].set_title("Reward-aligned evaluation average")
    axes[1].set_xlabel("Steps from event")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=True)
    fig.suptitle("Evaluation-average PETH: Normal vs Parkinson", fontsize=13)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(args.out, dpi=150)
    plt.close(fig)

    print(f"Saved evaluation-average PETH comparison -> {args.out}")


if __name__ == "__main__":
    main()
