"""
Evaluation script using the Parkinson dopamine variant agent.

This script is intended to load checkpoints trained with the normal
``a2c_rpe_model`` and evaluate them without retraining. Weights are NEVER
updated during this script — it is the "offline" evaluation counterpart to
evaluate_parkinsons_online.py, which continues learning during evaluation.

Usage
-----
    python evaluate_parkinsons.py                                 # uses checkpoints/a2c_rpe_final.pt
    python evaluate_parkinsons.py --checkpoint checkpoints/a2c_rpe_final.pt
    python evaluate_parkinsons.py --checkpoint checkpoints/a2c_rpe_final.pt --render --episodes 1000
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys


def _ensure_evaluation_runtime() -> None:
    """Re-launch with workspace venv when core dependencies are missing."""
    required = ("numpy", "torch")
    if all(importlib.util.find_spec(name) is not None for name in required):
        return

    if os.environ.get("HB_EVAL_REEXEC") == "1":
        return

    repo_root = os.path.dirname(os.path.abspath(__file__))
    venv_python = os.path.join(repo_root, ".venv", "bin", "python")
    if os.path.exists(venv_python):
        env = os.environ.copy()
        env["HB_EVAL_REEXEC"] = "1"
        os.execve(venv_python, [venv_python, os.path.abspath(__file__), *sys.argv[1:]], env)


_ensure_evaluation_runtime()

import numpy as np
import torch

from parkinsons_a2c_rpe_model import A2CAgent, parkinsons_rpe
from pick_and_place_env import PickAndPlaceEnv
from results import generate_plots_from_metrics
from logit_utils import get_motivation_updated_action_logits, assert_logits_match_forward


def _started_on_time(info: dict, grace_steps: int = 2) -> bool:
    """Return True when START occurs within the cue window plus grace."""
    if not bool(info.get("task_started", False)):
        return False

    start_step = info.get("task_started_step")
    cue_step = info.get("cue_step")
    start_window = info.get("start_window")

    if start_step is None or cue_step is None:
        return bool(info.get("task_started", False))

    if start_window is None:
        deadline = int(cue_step)
    else:
        deadline = int(cue_step) + max(1, int(start_window)) - 1

    deadline += max(0, int(grace_steps))

    return int(start_step) <= deadline


def _is_timed_success(info: dict, episode_length: int, success_time_limit: int) -> bool:
    """Return True only when the task is completed within the step limit."""
    return bool(info.get("object_placed", False)) and int(episode_length) <= int(success_time_limit)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a normal-trained checkpoint with the Parkinson dopamine variant agent."
    )
    parser.add_argument("--checkpoint", type=str, default="checkpoints/a2c_rpe_final.pt",
                        help="Path to a .pt checkpoint file.")
    parser.add_argument(
        "--agent_variant",
        type=str,
        default="parkinsons",
        choices=(
            "parkinsons",
            "parkinsons_no_shaping",
            "parkinsons_ldopa",
            "parkinsons_ldopa_no_shaping",
            "parkinsons_zero_rpe",
            "parkinsons_zero_rpe_no_shaping",
        ),
        help=(
            "Evaluation-only Parkinson mode: 'parkinsons' uses partial RPE transmission; "
            "'parkinsons_zero_rpe' forces zero RPE. Use *_no_shaping variants to "
            "disable movement reward shaping."
        ),
    )
    parser.add_argument("--grid_size",  type=int, default=5)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--episodes",   type=int, default=1000)
    parser.add_argument(
        "--prune_interval_episodes",
        type=int,
        default=30,
        help=(
            "Episodes between motivation-neuron pruning steps during evaluation. "
            "Higher values prune more slowly."
        ),
    )
    parser.add_argument(
        "--prune_neurons_per_interval",
        type=int,
        default=6,
        help="Number of motivation neurons pruned at each pruning step.",
    )
    parser.add_argument("--success_time_limit", type=int, default=75,
                        help="Count success only when placement occurs within this many steps.")
    parser.add_argument(
        "--start_grace_steps",
        type=int,
        default=2,
        help="Extra steps allowed beyond the strict cue window for counting START as on-time.",
    )
    parser.add_argument("--render",     action="store_true",
                        help="Print ASCII grid after each step.")
    parser.add_argument("--seed",       type=int, default=0)
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Directory to store evaluation metrics and plots.")
    parser.add_argument("--metrics_filename", type=str, default="evaluation_parkinsons_metrics.csv",
                        help="CSV filename for per-episode evaluation metrics.")
    parser.add_argument("--success_plot_filename", type=str, default="parkinsons_success_rate.png",
                        help="Filename for evaluation success-rate plot.")
    parser.add_argument("--reward_plot_filename", type=str, default="parkinsons_reward.png",
                        help="Filename for evaluation reward plot.")
    return parser.parse_args()


def _resolve_checkpoint_path(path: str) -> str:
    """Resolve checkpoint path, falling back to checkpoints/<basename>."""
    if os.path.exists(path):
        return path

    candidate = os.path.join("checkpoints", os.path.basename(path))
    if os.path.exists(candidate):
        return candidate

    raise FileNotFoundError(path)


def save_evaluation_metrics(path: str, rows: list[dict]) -> None:
    """Save per-episode evaluation metrics to CSV."""
    if not rows:
        return

    fieldnames = [
        "episode",
        "reward",
        "episode_length",
        "completed_task",
        "success",
        "started_task",
        "started_on_time",
        "cumulative_success_rate",
        "rolling_success_rate",
        "cumulative_start_rate",
        "rolling_start_rate",
        "tonic_dopamine",
        "mean_rpe",
        "mean_abs_rpe",
        "max_logit",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(args: argparse.Namespace) -> dict:
    """Run the agent for ``args.episodes`` episodes and return summary stats."""
    checkpoint_path = _resolve_checkpoint_path(args.checkpoint)
    base_variant = args.agent_variant.replace("_no_shaping", "")
    env_kwargs = {
        "grid_size": args.grid_size,
        "max_steps": 200,
        "shaping_start": 0.0,
        "shaping_end": 0.0,
    }
    env = PickAndPlaceEnv(**env_kwargs)
    # Evaluation-only reward policy: only PICK/PLACE produce non-zero rewards.
    env.reward_step = 0.0
    env.reward_invalid = 0.0
    env.reward_start = 0.0
    env.shaping_scale = 0.0
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent_kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "hidden_dim": args.hidden_dim,
        "prune_interval_episodes": args.prune_interval_episodes,
        "prune_neurons_per_interval": args.prune_neurons_per_interval,
    }
    if base_variant == "parkinsons_zero_rpe":
        # Evaluation-only severe impairment: no transmitted RPE.
        agent_kwargs.update(
            {
                "initial_surviving_fraction": 0.0,
                "initial_transmission_probability": 0.0,
            }
        )
    if base_variant == "parkinsons_ldopa":
        agent_kwargs["ldopa_compensation"] = True
    agent = A2CAgent(**agent_kwargs)

    if checkpoint_path:
        agent.load(checkpoint_path)
        print(f"Loaded checkpoint (trained with normal A2C RPE): {checkpoint_path}")
        print(f"Evaluation Parkinson variant: {args.agent_variant}")
        print("Reward shaping: disabled")
        print("Evaluation rewards: PICK/PLACE only (step/invalid/start = 0)")
        print(
            "Pruning schedule: "
            f"every {args.prune_interval_episodes} episodes, "
            f"{args.prune_neurons_per_interval} neurons per step"
        )
    else:
        print("No checkpoint provided – using randomly initialised weights.")

    rewards, lengths, successes, starts = [], [], [], []
    rows: list[dict] = []
    checked_logits_once = False

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        ep_reward = 0.0
        ep_length = 0
        terminated = truncated = False
        episode_max_logit = -float("inf")

        while not (terminated or truncated):
            if args.render:
                env.render()

            state_before = obs
            action, _ = agent.select_action(state_before)
            obs, reward, terminated, truncated, info = env.step(action)

            # Evaluation-only dopamine trace: compute Parkinson RPE signal,
            # and also log the motivation-updated max logit for this step.
            with torch.no_grad():
                state_t = torch.as_tensor(state_before, dtype=torch.float32).unsqueeze(0)
                next_state_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                _, value_t = agent.network(state_t)
                _, next_value_t = agent.network(next_state_t)
                done_flag = float(terminated or truncated)
                next_value_masked = next_value_t.squeeze(0).squeeze(-1) * (1.0 - done_flag)
                pd_delta = parkinsons_rpe(
                    reward=torch.tensor(float(reward), dtype=torch.float32),
                    gamma=agent.gamma,
                    value=value_t.squeeze(0).squeeze(-1),
                    next_value=next_value_masked,
                    initial_surviving_fraction=agent.surviving_fraction,
                    initial_transmission_probability=agent.transmission_probability,
                    current_episode=agent.current_episode,
                )

                if not checked_logits_once:
                    assert_logits_match_forward(agent.network, state_t)
                    checked_logits_once = True
                step_logits = get_motivation_updated_action_logits(agent.network, state_t)
                episode_max_logit = max(episode_max_logit, float(step_logits.max().item()))

            agent.dopamine.update(float(pd_delta.item()))

            ep_reward += reward
            ep_length += 1

        if args.render:
            env.render()
            print(f"--- Episode {ep+1} end. Reward={ep_reward:.2f} ---\n")

        rewards.append(ep_reward)
        lengths.append(ep_length)
        completed_task = bool(info.get("object_placed", False))
        timed_success = _is_timed_success(info, ep_length, args.success_time_limit)
        successes.append(timed_success)
        starts.append(_started_on_time(info, grace_steps=args.start_grace_steps))

        if hasattr(agent, "on_episode_end"):
            agent.on_episode_end()

        agent.current_episode += 1

        cumulative_success_rate = float(np.mean(successes))
        cumulative_start_rate = float(np.mean(starts))
        rolling_start_rate = float(np.mean(starts[max(0, len(starts) - 50):]))
        da = agent.dopamine.get_stats()
        rows.append(
            {
                "episode": ep + 1,
                "reward": float(ep_reward),
                "episode_length": int(ep_length),
                "completed_task": int(completed_task),
                "success": int(successes[-1]),
                "started_task": int(bool(info.get("task_started", False))),
                "started_on_time": int(starts[-1]),
                "cumulative_success_rate": cumulative_success_rate,
                "rolling_success_rate": cumulative_success_rate,
                "cumulative_start_rate": cumulative_start_rate,
                "rolling_start_rate": rolling_start_rate,
                "tonic_dopamine": float(da.get("tonic_level", 0.0)),
                "mean_rpe": float(da.get("mean_rpe", 0.0)),
                "mean_abs_rpe": float(da.get("mean_abs_rpe", 0.0)),
                "max_logit": float(episode_max_logit),
            }
        )

    stats = {
        "mean_reward": float(np.mean(rewards)),
        "std_reward":  float(np.std(rewards)),
        "mean_length": float(np.mean(lengths)),
        "success_rate": float(np.mean(successes)),
        "start_rate": float(np.mean(starts)),
        "episodes": args.episodes,
    }

    print(f"\nEvaluation over {args.episodes} episodes:")
    print(f"  Success criterion: placement within {args.success_time_limit} steps")
    print(f"  Mean reward  : {stats['mean_reward']:.3f} ± {stats['std_reward']:.3f}")
    print(f"  Mean length  : {stats['mean_length']:.1f}")
    print(f"  Timed success rate : {stats['success_rate']:.3f}")
    print(
        f"  Start rate   : {stats['start_rate']:.3f} "
        f"(strict + {args.start_grace_steps} grace steps)"
    )

    os.makedirs(args.results_dir, exist_ok=True)
    metrics_path = os.path.join(args.results_dir, args.metrics_filename)
    save_evaluation_metrics(metrics_path, rows)
    print(f"  Evaluation metrics saved → {metrics_path}")

    success_plot_path = os.path.join(args.results_dir, args.success_plot_filename)
    reward_plot_path = os.path.join(args.results_dir, args.reward_plot_filename)
    try:
        generate_plots_from_metrics(
            metrics_path=metrics_path,
            success_out=success_plot_path,
            reward_out=reward_plot_path,
            rolling_window=max(10, args.episodes // 10),
            title_prefix="Evaluation",
        )
        print(f"  Success-rate plot saved → {success_plot_path}")
        print(f"  Reward plot saved → {reward_plot_path}")
    except ModuleNotFoundError as exc:
        if exc.name == "matplotlib":
            print("  Plot generation skipped: matplotlib not available in this interpreter.")
            print("  Run plots with: .venv/bin/python results.py")
        else:
            raise

    return stats


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)