"""
Entry point for Layer 3 -- "Start on real arm".

Picks a trained model (Layer 1), rolls out one episode's worth of motor
vectors, and drives the physical servos with them (Layer 3). Isolated from
the sim: nothing here is imported by arm/sim.py, and a hardware failure
(no board, wrong port, SDK missing) prints a clear message and exits instead
of raising an unguarded exception.

Usage
-----
    .venv/bin/python run_arm_hardware.py --model healthy
    .venv/bin/python run_arm_hardware.py --model parkinsons --seed 3 --step_seconds 0.5

Runs on plain `python` (no GUI, no main-thread requirement like the sim).
"""

from __future__ import annotations

import argparse
import time

from arm.hardware import try_connect
from arm.model_bridge import VARIANTS, load_agent, rollout_episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Drive the real 3-DOF arm from a trained model.")
    parser.add_argument("--model", type=str, required=True, choices=VARIANTS)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid_size", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument(
        "--step_seconds", type=float, default=0.4,
        help="Real seconds to wait between motor commands (servos need real time to move).",
    )
    parser.add_argument("--port", type=str, default=None, help="Override serial port auto-detection.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading {args.model} agent and rolling out an episode (seed={args.seed})...")
    agent = load_agent(args.model, args.checkpoint)
    steps = rollout_episode(
        args.model,
        agent=agent,
        seed=args.seed,
        grid_size=args.grid_size,
        max_steps=args.max_steps,
    )
    print(f"Episode has {len(steps)} motor-vector steps.")

    hardware = try_connect(args.port)
    if hardware is None:
        print("No hardware connected -- nothing was sent to any servo. "
              "Plug in the URT board and try again.")
        return

    try:
        for step in steps:
            hardware.send_motor_values(step.motors)
            time.sleep(max(0.0, args.step_seconds))
        print("Episode complete.")
    except KeyboardInterrupt:
        print("Interrupted -- stopping.")
    finally:
        hardware.close()


if __name__ == "__main__":
    main()
