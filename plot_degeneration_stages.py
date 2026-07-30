"""
Start rate vs. success rate, one point per motivation-neuron degeneration stage.

The evaluation metrics CSVs record per-episode behaviour but not the number of
surviving motivation neurons, so the stage is reconstructed from the pruning
schedule in ``A2CAgent.on_episode_end`` (parkinsons_a2c_rpe_model.py).

Usage
-----
    python plot_degeneration_stages.py
    python plot_degeneration_stages.py --online_interval 30
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys


def _ensure_plot_runtime() -> None:
    """Re-launch with workspace venv when matplotlib is missing."""
    if importlib.util.find_spec("matplotlib") is not None:
        return

    if os.environ.get("HB_STAGES_REEXEC") == "1":
        return

    repo_root = os.path.dirname(os.path.abspath(__file__))
    venv_python = os.path.join(repo_root, ".venv", "bin", "python")
    if os.path.exists(venv_python):
        env = os.environ.copy()
        env["HB_STAGES_REEXEC"] = "1"
        os.execve(venv_python, [venv_python, os.path.abspath(__file__), *sys.argv[1:]], env)


_ensure_plot_runtime()


TOTAL_NEURONS = 128
PRUNED_PER_STEP = 6
MIN_NEURONS = 39  # ceil(128 * min_motivation_neuron_fraction=0.30)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument(
        "--offline_interval",
        type=int,
        default=30,
        help=(
            "Episodes between pruning steps in the offline run "
            "(evaluate_parkinsons.py --prune_interval_episodes default)."
        ),
    )
    parser.add_argument(
        "--online_interval",
        type=int,
        default=15,
        help=(
            "Episodes between pruning steps in the online run "
            "(A2CAgent default; evaluate_parkinsons_online.py never overrides it)."
        ),
    )
    parser.add_argument("--total_neurons", type=int, default=TOTAL_NEURONS)
    parser.add_argument("--pruned_per_step", type=int, default=PRUNED_PER_STEP)
    parser.add_argument("--min_neurons", type=int, default=MIN_NEURONS)
    return parser.parse_args()


def alive_neurons(
    episode: int,
    interval: int,
    total: int = TOTAL_NEURONS,
    per_step: int = PRUNED_PER_STEP,
    floor: int = MIN_NEURONS,
) -> int:
    """Surviving motivation neurons during 1-indexed ``episode``.

    ``on_episode_end`` prunes *after* the episode's metrics row is written, so
    episode ``ep`` runs with the count established by the preceding ``ep - 1``
    episodes.
    """
    steps = (max(1, int(episode)) - 1) // max(1, int(interval))
    return max(floor, total - per_step * steps)


def stage_averages(csv_path: str, interval: int, **schedule) -> list[dict]:
    """Average start rate and success rate over each degeneration stage."""
    groups: dict[int, list[dict]] = {}
    with open(csv_path, newline="", encoding="utf-8") as fp:
        for row in csv.DictReader(fp):
            alive = alive_neurons(int(row["episode"]), interval, **schedule)
            groups.setdefault(alive, []).append(row)

    stages = []
    for alive in sorted(groups, reverse=True):
        rows = groups[alive]
        stages.append(
            {
                "alive_neurons": alive,
                "start_rate": sum(int(r["started_on_time"]) for r in rows) / len(rows),
                "success_rate": sum(int(r["success"]) for r in rows) / len(rows),
                "n_episodes": len(rows),
                "first_episode": min(int(r["episode"]) for r in rows),
                "last_episode": max(int(r["episode"]) for r in rows),
            }
        )
    return stages


def plot_stages(
    stages: list[dict],
    out_path: str,
    title: str,
    vmin: int = MIN_NEURONS,
    vmax: int = TOTAL_NEURONS,
) -> None:
    mpl_config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mplconfig")
    os.makedirs(mpl_config_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", mpl_config_dir)
    import matplotlib.pyplot as plt

    start = [s["start_rate"] for s in stages]
    success = [s["success_rate"] for s in stages]
    alive = [s["alive_neurons"] for s in stages]

    fig, ax = plt.subplots(1, 1, figsize=(7.6, 6.4))
    # Stages are ordered 128 -> 39, so the line traces the degeneration trajectory.
    ax.plot(start, success, color="0.6", linewidth=1.0, alpha=0.6, zorder=1)
    # coolwarm_r (not coolwarm) puts the high end -- 128 neurons -- at the cool
    # blue end and warms as neurons are lost.
    scatter = ax.scatter(
        start,
        success,
        c=alive,
        cmap="coolwarm_r",
        vmin=vmin,
        vmax=vmax,
        s=110,
        edgecolors="0.25",
        linewidths=0.6,
        zorder=2,
    )

    ax.set_xlabel("Start rate (task started on time)")
    ax.set_ylabel("Success rate (placed within time limit)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.grid(alpha=0.3)

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Surviving motivation neurons")
    cbar.set_ticks([s["alive_neurons"] for s in stages][::3])

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_stage_table(path: str, tables: dict[str, list[dict]]) -> None:
    fieldnames = [
        "run",
        "alive_neurons",
        "start_rate",
        "success_rate",
        "n_episodes",
        "first_episode",
        "last_episode",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for run, stages in tables.items():
            for stage in stages:
                writer.writerow({"run": run, **stage})


def _print_table(run: str, interval: int, stages: list[dict]) -> None:
    total = sum(s["n_episodes"] for s in stages)
    print(f"\n{run}  (prune interval {interval} episodes, {len(stages)} stages, {total} episodes)")
    print("  neurons  episodes      n   start rate   success rate")
    print("  " + "-" * 52)
    for s in stages:
        span = f"{s['first_episode']}-{s['last_episode']}"
        print(
            f"  {s['alive_neurons']:7d}  {span:>9s}  {s['n_episodes']:5d}"
            f"   {s['start_rate']:10.3f}   {s['success_rate']:12.3f}"
        )


def main() -> None:
    args = parse_args()
    schedule = {
        "total": args.total_neurons,
        "per_step": args.pruned_per_step,
        "floor": args.min_neurons,
    }

    runs = [
        (
            "parkinsons",
            "evaluation_parkinsons_metrics.csv",
            args.offline_interval,
            "parkinsons_start_vs_success_by_stage.png",
            "Parkinson's evaluation: start vs. success by degeneration stage",
        ),
        (
            "parkinsons_online",
            "evaluation_parkinsons_online_metrics.csv",
            args.online_interval,
            "parkinsons_online_start_vs_success_by_stage.png",
            "Parkinson's online evaluation: start vs. success by degeneration stage",
        ),
    ]

    tables: dict[str, list[dict]] = {}
    for run, metrics_name, interval, plot_name, title in runs:
        metrics_path = os.path.join(args.results_dir, metrics_name)
        if not os.path.exists(metrics_path):
            raise SystemExit(f"Missing evaluation metrics: {metrics_path}")

        stages = stage_averages(metrics_path, interval, **schedule)
        tables[run] = stages
        _print_table(run, interval, stages)

        out_path = os.path.join(args.results_dir, plot_name)
        plot_stages(
            stages,
            out_path,
            title,
            vmin=args.min_neurons,
            vmax=args.total_neurons,
        )
        print(f"  plot saved -> {os.path.abspath(out_path)}")

    table_path = os.path.join(args.results_dir, "start_vs_success_by_stage.csv")
    save_stage_table(table_path, tables)
    print(f"\nstage table saved -> {os.path.abspath(table_path)}")
    print(
        "\nNote: the floor stage (39 neurons) averages far more episodes than the\n"
        "other stages -- see the n_episodes column before comparing point to point."
    )


if __name__ == "__main__":
    main()
