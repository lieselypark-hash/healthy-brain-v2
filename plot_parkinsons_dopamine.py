"""
Plot tonic dopamine traces for Parkinson and Normal runs.

Usage
-----
python plot_parkinsons_dopamine.py
python plot_parkinsons_dopamine.py \
    --parkinsons_input results/evaluation_parkinsons_metrics.csv \
    --normal_input results/evaluation_normal_metrics.csv
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
        description=(
            "Plot tonic dopamine for Parkinson and Normal runs, and an overlay "
            "comparison on shared axes."
        )
    )
    parser.add_argument(
        "--parkinsons_input",
        type=str,
        default="results/evaluation_parkinsons_metrics.csv",
        help="Parkinson metrics CSV with episode and tonic_dopamine columns.",
    )
    parser.add_argument(
        "--normal_input",
        type=str,
        default="results/evaluation_normal_metrics.csv",
        help="Normal metrics CSV with episode and tonic_dopamine columns.",
    )
    parser.add_argument(
        "--parkinsons_output",
        type=str,
        default="results/parkinsons_dopamine.png",
        help="Output PNG path for Parkinson-only dopamine plot.",
    )
    parser.add_argument(
        "--normal_output",
        type=str,
        default="results/normal_dopamine.png",
        help="Output PNG path for Normal-only dopamine plot.",
    )
    parser.add_argument(
        "--overlay_output",
        type=str,
        default="results/normal_vs_parkinsons_dopamine_overlay.png",
        help="Output PNG path for overlaid Normal vs Parkinson dopamine plot.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=170,
        help="Output image DPI.",
    )
    return parser.parse_args()


def _read_metrics(path: str) -> tuple[list[int], list[float]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metrics file not found: {path}")

    episodes: list[int] = []
    tonic: list[float] = []

    with open(path, newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        required = {"episode", "tonic_dopamine"}
        missing = required.difference(set(reader.fieldnames or []))
        if missing:
            raise ValueError(
                "Input CSV is missing required columns: "
                + ", ".join(sorted(missing))
            )

        for row in reader:
            episodes.append(int(row["episode"]))
            tonic.append(float(row["tonic_dopamine"]))

    if not episodes:
        raise ValueError(f"Input CSV has no rows: {path}")

    return episodes, tonic


def _save_fig(fig: plt.Figure, out_path: str, dpi: int) -> None:
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_single_dopamine(
    episodes: list[int],
    tonic: list[float],
    label: str,
    color: str,
    out_path: str,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(10.8, 4.8))
    ax.plot(episodes, tonic, linewidth=2.0, color=color, label=label)
    ax.axhline(0.0, linewidth=1.0, linestyle="--", alpha=0.5, color="black")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Tonic dopamine")
    ax.set_title(f"{label} Tonic Dopamine")
    ax.grid(alpha=0.25)
    ax.legend()
    _save_fig(fig, out_path, dpi)


def plot_overlay_dopamine(
    pd_episodes: list[int],
    pd_tonic: list[float],
    normal_episodes: list[int],
    normal_tonic: list[float],
    out_path: str,
    dpi: int,
) -> None:
    shared_len = min(len(pd_episodes), len(normal_episodes))
    if shared_len == 0:
        raise ValueError("Cannot overlay dopamine traces: one of the series is empty.")

    pd_episodes = pd_episodes[:shared_len]
    pd_tonic = pd_tonic[:shared_len]
    normal_episodes = normal_episodes[:shared_len]
    normal_tonic = normal_tonic[:shared_len]

    fig, ax = plt.subplots(1, 1, figsize=(10.8, 4.8))
    ax.plot(normal_episodes, normal_tonic, linewidth=2.0, color="#118ab2", label="Normal")
    ax.plot(pd_episodes, pd_tonic, linewidth=2.0, color="#d1495b", label="Parkinson")
    ax.axhline(0.0, linewidth=1.0, linestyle="--", alpha=0.5, color="black")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Tonic dopamine")
    ax.set_title("Tonic Dopamine: Normal vs Parkinson (Overlay)")
    ax.grid(alpha=0.25)
    ax.legend()
    _save_fig(fig, out_path, dpi)


def main() -> None:
    args = parse_args()
    pd_episodes, pd_tonic = _read_metrics(args.parkinsons_input)
    normal_episodes, normal_tonic = _read_metrics(args.normal_input)

    plot_single_dopamine(
        episodes=pd_episodes,
        tonic=pd_tonic,
        label="Parkinson",
        color="#d1495b",
        out_path=args.parkinsons_output,
        dpi=args.dpi,
    )
    plot_single_dopamine(
        episodes=normal_episodes,
        tonic=normal_tonic,
        label="Normal",
        color="#118ab2",
        out_path=args.normal_output,
        dpi=args.dpi,
    )
    plot_overlay_dopamine(
        pd_episodes=pd_episodes,
        pd_tonic=pd_tonic,
        normal_episodes=normal_episodes,
        normal_tonic=normal_tonic,
        out_path=args.overlay_output,
        dpi=args.dpi,
    )
    print(f"Saved Parkinson dopamine plot -> {args.parkinsons_output}")
    print(f"Saved Normal dopamine plot -> {args.normal_output}")
    print(f"Saved overlay dopamine plot -> {args.overlay_output}")


if __name__ == "__main__":
    main()
