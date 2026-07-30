"""
Entry point for Layer 2 -- ties the model (Layer 1) to the MuJoCo sim (Layer 2).

This is the only place that imports both `arm.model_bridge` (torch-based RL
agents) and `arm.sim` (MuJoCo). `arm/sim.py` itself never imports the model
code, so the sim keeps working even if the model changes completely.

Usage
-----
    .venv/bin/mjpython run_arm_sim.py                       # both arms, seed 0
    .venv/bin/mjpython run_arm_sim.py --models healthy
    .venv/bin/mjpython run_arm_sim.py --models parkinsons --seed 3

macOS requires `mjpython` (not plain `python`) for the native viewer window,
since MuJoCo's GUI must run on the main thread.
"""

from __future__ import annotations

import argparse

from arm.model_bridge import VARIANTS, load_agent, rollout_episode
from arm.sim import ArmSimulator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Side-by-side 3-DOF arm sim viewer.")
    parser.add_argument(
        "--models",
        type=str,
        default="healthy,parkinsons",
        help=f"Comma-separated subset of {VARIANTS} to load.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Initial episode seed.")
    parser.add_argument("--grid_size", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--healthy_checkpoint", type=str, default=None)
    parser.add_argument("--parkinsons_checkpoint", type=str, default=None)
    parser.add_argument(
        "--no_tremor", action="store_true",
        help="Disable the cosmetic Parkinsonian resting-tremor jitter in the viewer.",
    )
    parser.add_argument(
        "--no_auto_replay", action="store_true",
        help="Don't automatically start a new episode when the current one finishes.",
    )
    parser.add_argument(
        "--replay_hold_seconds", type=float, default=1.5,
        help="Seconds to hold on the final frame before auto-replaying.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = tuple(m.strip() for m in args.models.split(",") if m.strip())
    for label in labels:
        if label not in VARIANTS:
            raise SystemExit(f"Unknown model {label!r}; choose from {VARIANTS}")

    checkpoints = {
        "healthy": args.healthy_checkpoint,
        "parkinsons": args.parkinsons_checkpoint,
    }

    print("Loading agent checkpoints...")
    agents = {
        label: load_agent(label, checkpoints[label])
        for label in labels
    }

    def on_request_episode(label: str, seed: int):
        steps = rollout_episode(
            label,
            agent=agents[label],
            seed=seed,
            grid_size=args.grid_size,
            max_steps=args.max_steps,
        )
        return [step.motors for step in steps]

    sim = ArmSimulator(
        labels=labels,
        on_request_episode=on_request_episode,
        enable_tremor=not args.no_tremor,
        auto_replay=not args.no_auto_replay,
        replay_hold_seconds=args.replay_hold_seconds,
    )
    for label in labels:
        sim.set_trajectory(label, on_request_episode(label, args.seed))

    sim.run()


if __name__ == "__main__":
    main()
