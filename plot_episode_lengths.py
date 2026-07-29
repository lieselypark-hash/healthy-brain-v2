"""
Plot evaluation episode lengths (time steps taken per episode) for Normal,
Parkinsons, and Parkinsons Online evaluations.

Usage
-----
python plot_episode_lengths.py
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys


def _ensure_plot_runtime() -> None:
    """Re-launch with workspace venv if plotting deps are missing."""
    needed = ("matplotlib",)
    if all(importlib.util.find_spec(name) is not None for name in needed):
        return

    if os.environ.get("HB_PLOT_REEXEC") == "1":
        return

    repo_root = os.path.dirname(os.path.abspath(__file__))
    venv_python = os.path.join(repo_root, ".venv", "bin", "python")
    if os.path.exists(venv_python):
        env = os.environ.copy()
        env["HB_PLOT_REEXEC"] = "1"
        os.execve(venv_python, [venv_python, os.path.abspath(__file__), *sys.argv[1:]], env)


_ensure_plot_runtime()

_MPL_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mplconfig")
os.makedirs(_MPL_CONFIG_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _MPL_CONFIG_DIR)

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot time steps taken per episode for normal, parkinsons, and parkinsons online evaluations."
    )
    parser.add_argument("--normal_input", type=str, default="results/evaluation_normal_metrics.csv")
    parser.add_argument("--parkinsons_input", type=str, default="results/evaluation_parkinsons_metrics.csv")
    parser.add_argument("--parkinsons_online_input", type=str, default="results/evaluation_parkinsons_online_metrics.csv")

    parser.add_argument("--normal_output", type=str, default="results/normal_episode_lengths.png")
    parser.add_argument("--parkinsons_output", type=str, default="results/parkinsons_episode_lengths.png")
    parser.add_argument("--parkinsons_online_output", type=str, default="results/parkinsons_online_episode_lengths.png")
    parser.add_argument("--overlay_output", type=str, default="results/evaluation_episode_lengths_overlay.png")
    parser.add_argument("--bin_size", type=int, default=10,
                        help="Number of episodes per averaging bin.")
    parser.add_argument("--dpi", type=int, default=170)
    return parser.parse_args()


def _read_episode_lengths(path: str) -> tuple[list[int], list[int]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metrics file not found: {path}")

    episodes: list[int] = []
    lengths: list[int] = []

    with open(path, newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        required = {"episode", "episode_length"}
        missing = required.difference(set(reader.fieldnames or []))
        if missing:
            raise ValueError(
                "Input CSV is missing required columns: " + ", ".join(sorted(missing))
            )

        for row in reader:
            episodes.append(int(float(row["episode"])))
            lengths.append(int(float(row["episode_length"])))

    if not episodes:
        raise ValueError(f"Input CSV has no rows: {path}")

    return episodes, lengths


def _save_figure(fig: plt.Figure, out_path: str, dpi: int) -> None:
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _bin_average(episodes: list[int], lengths: list[int], bin_size: int) -> tuple[list[int], list[float]]:
    if bin_size <= 0:
        raise ValueError("bin_size must be > 0")

    binned_episodes: list[int] = []
    binned_means: list[float] = []

    for i in range(0, len(episodes), bin_size):
        ep_chunk = episodes[i:i + bin_size]
        len_chunk = lengths[i:i + bin_size]
        if not ep_chunk:
            continue
        binned_episodes.append(ep_chunk[-1])
        binned_means.append(float(sum(len_chunk) / len(len_chunk)))

    return binned_episodes, binned_means


def _plot_single(
    episodes: list[int],
    lengths: list[int],
    label: str,
    color: str,
    out_path: str,
    dpi: int,
    bin_size: int,
) -> None:
    x_vals, y_vals = _bin_average(episodes, lengths, bin_size)
    fig, ax = plt.subplots(1, 1, figsize=(10.8, 4.8))
    ax.plot(x_vals, y_vals, linewidth=2.0, color=color, label=label)
    ax.set_xlabel("Evaluation episode")
    ax.set_ylabel(f"Average time steps (per {bin_size} episodes)")
    ax.set_title(f"{label}: Average Episode Length per {bin_size} Episodes")
    ax.grid(alpha=0.25)
    ax.legend()
    _save_figure(fig, out_path, dpi)


def _plot_overlay(
    normal_ep: list[int],
    normal_len: list[int],
    pd_ep: list[int],
    pd_len: list[int],
    pd_online_ep: list[int],
    pd_online_len: list[int],
    out_path: str,
    dpi: int,
    bin_size: int,
) -> None:
    normal_x, normal_y = _bin_average(normal_ep, normal_len, bin_size)
    pd_x, pd_y = _bin_average(pd_ep, pd_len, bin_size)
    pd_online_x, pd_online_y = _bin_average(pd_online_ep, pd_online_len, bin_size)

    fig, ax = plt.subplots(1, 1, figsize=(10.8, 4.8))
    ax.plot(normal_x, normal_y, linewidth=2.0, color="#118ab2", label="Normal")
    ax.plot(pd_x, pd_y, linewidth=2.0, color="#d1495b", label="Parkinson")
    ax.plot(pd_online_x, pd_online_y, linewidth=2.0, color="#2a9d8f", label="Parkinson Online")
    ax.set_xlabel("Evaluation episode")
    ax.set_ylabel(f"Average time steps (per {bin_size} episodes)")
    ax.set_title(
        f"Average Episode Length per {bin_size} Episodes: "
        "Normal vs Parkinson vs Parkinson Online"
    )
    ax.grid(alpha=0.25)
    ax.legend()
    _save_figure(fig, out_path, dpi)


def main() -> None:
    args = parse_args()

    normal_ep, normal_len = _read_episode_lengths(args.normal_input)
    pd_ep, pd_len = _read_episode_lengths(args.parkinsons_input)
    pd_online_ep, pd_online_len = _read_episode_lengths(args.parkinsons_online_input)

    _plot_single(
        normal_ep,
        normal_len,
        "Normal",
        "#118ab2",
        args.normal_output,
        args.dpi,
        args.bin_size,
    )
    _plot_single(
        pd_ep,
        pd_len,
        "Parkinson",
        "#d1495b",
        args.parkinsons_output,
        args.dpi,
        args.bin_size,
    )
    _plot_single(
        pd_online_ep,
        pd_online_len,
        "Parkinson Online",
        "#2a9d8f",
        args.parkinsons_online_output,
        args.dpi,
        args.bin_size,
    )

    _plot_overlay(
        normal_ep,
        normal_len,
        pd_ep,
        pd_len,
        pd_online_ep,
        pd_online_len,
        args.overlay_output,
        args.dpi,
        args.bin_size,
    )

    print(f"Saved normal episode-length plot -> {args.normal_output}")
    print(f"Saved parkinsons episode-length plot -> {args.parkinsons_output}")
    print(f"Saved parkinsons online episode-length plot -> {args.parkinsons_online_output}")
    print(f"Saved overlaid episode-length plot -> {args.overlay_output}")


if __name__ == "__main__":
    main()
