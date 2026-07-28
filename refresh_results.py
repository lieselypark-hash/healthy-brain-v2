"""
Regenerate all result graphs in one command.

Usage
-----
python refresh_results.py
python refresh_results.py --no_peth
"""

from __future__ import annotations

import argparse
import os

from results import (
    ensure_plot_runtime,
    generate_evaluation_comparison_plots,
    generate_peth_outputs,
    generate_plots_from_metrics,
)


LEGACY_FILES = [
    "results/training_results.png",
    "results/evaluation_metrics.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean stale artifacts and regenerate all results plots."
    )
    parser.add_argument("--no_peth", action="store_true", help="Skip PETH plot generation.")
    parser.add_argument("--rolling_window", type=int, default=50, help="Rolling window for reward plots.")
    parser.add_argument(
        "--success_time_limit",
        type=int,
        default=75,
        help="Count success only when completion occurs within this many steps.",
    )
    parser.add_argument("--peth_probe_episodes", type=int, default=200)
    parser.add_argument("--peth_window_pre", type=int, default=10)
    parser.add_argument("--peth_window_post", type=int, default=20)
    return parser.parse_args()


def _remove_if_exists(path: str) -> bool:
    if os.path.exists(path):
        os.remove(path)
        return True
    return False


def _require(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required metrics file not found: {path}")


def main() -> None:
    ensure_plot_runtime()
    args = parse_args()

    os.makedirs("results", exist_ok=True)

    removed = [path for path in LEGACY_FILES if _remove_if_exists(path)]
    if removed:
        for path in removed:
            print(f"Removed legacy artifact -> {path}")

    _require("results/training_metrics.csv")
    _require("results/evaluation_normal_metrics.csv")
    _require("results/evaluation_parkinsons_metrics.csv")

    training_success, training_reward = generate_plots_from_metrics(
        metrics_path="results/training_metrics.csv",
        success_out="results/success_rate.png",
        reward_out="results/reward.png",
        rolling_window=max(1, args.rolling_window),
        title_prefix="Training",
        success_time_limit=args.success_time_limit,
    )
    print(f"Saved training success plot -> {training_success}")
    print(f"Saved training reward plot -> {training_reward}")

    normal_success, normal_reward = generate_plots_from_metrics(
        metrics_path="results/evaluation_normal_metrics.csv",
        success_out="results/normal_success_rate.png",
        reward_out="results/normal_reward.png",
        rolling_window=max(1, args.rolling_window),
        title_prefix="Evaluation (Normal)",
        success_time_limit=args.success_time_limit,
    )
    print(f"Saved normal success plot -> {normal_success}")
    print(f"Saved normal reward plot -> {normal_reward}")

    pd_success, pd_reward = generate_plots_from_metrics(
        metrics_path="results/evaluation_parkinsons_metrics.csv",
        success_out="results/parkinsons_success_rate.png",
        reward_out="results/parkinsons_reward.png",
        rolling_window=max(1, args.rolling_window),
        title_prefix="Evaluation (Parkinson)",
        success_time_limit=args.success_time_limit,
    )
    print(f"Saved Parkinson success plot -> {pd_success}")
    print(f"Saved Parkinson reward plot -> {pd_reward}")

    compare_paths = generate_evaluation_comparison_plots(
        normal_metrics_path="results/evaluation_normal_metrics.csv",
        parkinsons_metrics_path="results/evaluation_parkinsons_metrics.csv",
        success_out="results/evaluation_success_comparison.png",
        reward_out="results/evaluation_reward_comparison.png",
        rolling_window=max(10, args.rolling_window // 2),
        success_time_limit=args.success_time_limit,
    )
    if compare_paths is not None:
        print(f"Saved evaluation success comparison -> {compare_paths[0]}")
        print(f"Saved evaluation reward comparison -> {compare_paths[1]}")

    if not args.no_peth:
        generate_peth_outputs(
            probe_episodes=max(1, args.peth_probe_episodes),
            window_pre=max(0, args.peth_window_pre),
            window_post=max(1, args.peth_window_post),
        )
        print("Saved PETH plots and event-count CSV files.")
        print("Saved RPE component figures.")


if __name__ == "__main__":
    main()