"""
Evaluation script using the Parkinson dopamine variant agent with online updates.

Unlike evaluate_parkinsons.py, this script continues learning during evaluation
(online adaptation) while initializing from a trained checkpoint.

Usage
-----
    python evaluate_parkinsons_online.py
    python evaluate_parkinsons_online.py --checkpoint checkpoints/a2c_rpe_final.pt
    python evaluate_parkinsons_online.py --episodes 1000 --n_steps 10
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
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

from parkinsons_a2c_rpe_model import A2CAgent
from pick_and_place_env import PickAndPlaceEnv
from results import generate_plots_from_metrics
from logit_utils import get_motivation_updated_action_logits, assert_logits_match_forward


def _started_on_time(info: dict) -> bool:
    """Return True only when START occurred within the allowed cue window."""
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

    return int(start_step) <= deadline


def _is_timed_success(info: dict, episode_length: int, success_time_limit: int) -> bool:
    """Return True only when the task is completed within the step limit."""
    return bool(info.get("object_placed", False)) and int(episode_length) <= int(success_time_limit)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a normal-trained checkpoint with the Parkinson dopamine variant agent "
            "while continuing to update weights online."
        )
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
            "parkinsons_progressive_pruning",
            "parkinsons_progressive_pruning_no_shaping",
            "parkinsons_ldopa",
            "parkinsons_ldopa_no_shaping",
            "parkinsons_zero_rpe",
            "parkinsons_zero_rpe_no_shaping",
        ),
        help=(
            "Parkinson mode: 'parkinsons' uses partial RPE transmission; "
            "'parkinsons_progressive_pruning' matches offline-style episode-wise motivation pruning; "
            "'parkinsons_zero_rpe' forces zero RPE. Use *_no_shaping variants to "
            "disable movement reward shaping."
        ),
    )
    parser.add_argument("--grid_size", type=int, default=5)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--n_steps", type=int, default=10,
                        help="Number of environment steps per online update.")
    parser.add_argument("--success_time_limit", type=int, default=75,
                        help="Count success only when placement occurs within this many steps.")
    parser.add_argument("--render", action="store_true",
                        help="Print ASCII grid after each step.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Directory to store evaluation metrics and plots.")
    parser.add_argument("--metrics_filename", type=str, default="evaluation_parkinsons_online_metrics.csv",
                        help="CSV filename for per-episode evaluation metrics.")
    parser.add_argument("--success_plot_filename", type=str, default="parkinsons_online_success_rate.png",
                        help="Filename for evaluation success-rate plot.")
    parser.add_argument("--reward_plot_filename", type=str, default="parkinsons_online_reward.png",
                        help="Filename for evaluation reward plot.")
    parser.add_argument("--save_adapted_checkpoint", type=str, default="",
                        help="Optional output path to save post-evaluation adapted weights.")
    parser.add_argument(
        "--transitions_filename",
        type=str,
        default="evaluation_parkinsons_online_transitions.csv",
        help="CSV filename for per-step transitions collected during online evaluation.",
    )
    parser.add_argument(
        "--no_save_transitions",
        action="store_true",
        help="Disable saving per-step transitions.",
    )
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


def save_transitions(path: str, rows: list[dict]) -> None:
    """Save per-step transitions to CSV."""
    if not rows:
        return

    fieldnames = [
        "episode",
        "step",
        "state",
        "action",
        "action_prob",
        "reward",
        "next_state",
        "done",
        "terminated",
        "truncated",
        "cue_active",
        "task_started",
        "object_placed",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_online(args: argparse.Namespace) -> dict:
    """Run online-updating evaluation over ``args.episodes`` episodes."""
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
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent_kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "hidden_dim": args.hidden_dim,
    }
    if base_variant == "parkinsons_zero_rpe":
        agent_kwargs.update(
            {
                "surviving_fraction": 0.0,
                "transmission_probability": 0.0,
            }
        )
    if base_variant == "parkinsons_ldopa":
        agent_kwargs["ldopa_compensation"] = True
    progressive_pruning = base_variant == "parkinsons_progressive_pruning"

    agent = A2CAgent(**agent_kwargs)
    agent.load(checkpoint_path)
    # Default online mode keeps baseline impairment fixed;
    # progressive_pruning variant mirrors offline episode-wise pruning.
    if not progressive_pruning and hasattr(agent, "set_motivation_active_fraction"):
        agent.set_motivation_active_fraction(0.30)
    print(f"Loaded checkpoint (trained with normal A2C RPE): {checkpoint_path}")
    print(f"Evaluation Parkinson variant: {args.agent_variant}")
    print("Reward shaping: disabled")
    print("Evaluation rewards: PICK/PLACE only (step/invalid/start = 0)")
    print(f"Evaluation mode: online updates every {args.n_steps} steps")
    if progressive_pruning:
        print("Impairment schedule: progressive motivation-neuron pruning each episode")
    else:
        print("Baseline impairment: fixed 70% motivation-neuron disablement")

    rewards, lengths, successes, starts = [], [], [], []
    rows = []
    transition_rows = []
    checked_logits_once = False

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        ep_reward = 0.0
        ep_length = 0
        terminated = False
        truncated = False
        last_info = {}
        episode_max_logit = -float("inf")

        while not (terminated or truncated):
            batch_s, batch_a, batch_r, batch_ns, batch_d, batch_old_p = [], [], [], [], [], []

            for _ in range(args.n_steps):
                if args.render:
                    env.render()

                with torch.no_grad():
                    state_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                    if not checked_logits_once:
                        assert_logits_match_forward(agent.network, state_t)
                        checked_logits_once = True
                    step_logits = get_motivation_updated_action_logits(agent.network, state_t)
                    episode_max_logit = max(episode_max_logit, float(step_logits.max().item()))

                action, action_probs = agent.select_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)

                transition_rows.append(
                    {
                        "episode": int(ep + 1),
                        "step": int(ep_length + 1),
                        "state": json.dumps(np.asarray(obs, dtype=np.float32).tolist()),
                        "action": int(action),
                        "action_prob": float(action_probs[action].item()),
                        "reward": float(reward),
                        "next_state": json.dumps(np.asarray(next_obs, dtype=np.float32).tolist()),
                        "done": int(terminated or truncated),
                        "terminated": int(terminated),
                        "truncated": int(truncated),
                        "cue_active": int(bool(info.get("cue_active", False))),
                        "task_started": int(bool(info.get("task_started", False))),
                        "object_placed": int(bool(info.get("object_placed", False))),
                    }
                )

                batch_s.append(obs)
                batch_a.append(action)
                batch_r.append(float(reward))
                batch_ns.append(next_obs)
                batch_d.append(float(terminated or truncated))
                batch_old_p.append(float(action_probs[action].item()))

                ep_reward += float(reward)
                ep_length += 1
                obs = next_obs
                last_info = info

                if terminated or truncated:
                    break

            if batch_s:
                agent.update(batch_s, batch_a, batch_r, batch_ns, batch_d, batch_old_p)

        if args.render:
            env.render()
            print(f"--- Episode {ep+1} end. Reward={ep_reward:.2f} ---\n")

        completed_task = bool(last_info.get("object_placed", False))
        timed_success = _is_timed_success(last_info, ep_length, args.success_time_limit)
        starts_on_time = _started_on_time(last_info)

        rewards.append(ep_reward)
        lengths.append(ep_length)
        successes.append(timed_success)
        starts.append(starts_on_time)

        cumulative_success_rate = float(np.mean(successes))
        cumulative_start_rate = float(np.mean(starts))
        rolling_start_rate = float(np.mean(starts[max(0, len(starts) - 50):]))
        rolling_success_rate = float(np.mean(successes[max(0, len(successes) - 50):]))
        da = agent.dopamine.get_stats()

        rows.append(
            {
                "episode": ep + 1,
                "reward": float(ep_reward),
                "episode_length": int(ep_length),
                "completed_task": int(completed_task),
                "success": int(timed_success),
                "started_task": int(bool(last_info.get("task_started", False))),
                "started_on_time": int(starts_on_time),
                "cumulative_success_rate": cumulative_success_rate,
                "rolling_success_rate": rolling_success_rate,
                "cumulative_start_rate": cumulative_start_rate,
                "rolling_start_rate": rolling_start_rate,
                "tonic_dopamine": float(da.get("tonic_level", 0.0)),
                "mean_rpe": float(da.get("mean_rpe", 0.0)),
                "mean_abs_rpe": float(da.get("mean_abs_rpe", 0.0)),
                "max_logit": float(episode_max_logit),
            }
        )

        if progressive_pruning and hasattr(agent, "on_episode_end"):
            agent.on_episode_end()

    stats = {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_length": float(np.mean(lengths)),
        "success_rate": float(np.mean(successes)),
        "start_rate": float(np.mean(starts)),
        "episodes": args.episodes,
    }

    print(f"\nOnline evaluation over {args.episodes} episodes:")
    print(f"  Success criterion: placement within {args.success_time_limit} steps")
    print(f"  Mean reward  : {stats['mean_reward']:.3f} +- {stats['std_reward']:.3f}")
    print(f"  Mean length  : {stats['mean_length']:.1f}")
    print(f"  Timed success rate : {stats['success_rate']:.3f}")
    print(f"  Start rate   : {stats['start_rate']:.3f} (strict: on-time starts)")

    os.makedirs(args.results_dir, exist_ok=True)
    metrics_path = os.path.join(args.results_dir, args.metrics_filename)
    save_evaluation_metrics(metrics_path, rows)
    print(f"  Evaluation metrics saved -> {metrics_path}")

    if not args.no_save_transitions:
        transitions_path = os.path.join(args.results_dir, args.transitions_filename)
        save_transitions(transitions_path, transition_rows)
        print(f"  Transition log saved -> {transitions_path}")

    success_plot_path = os.path.join(args.results_dir, args.success_plot_filename)
    reward_plot_path = os.path.join(args.results_dir, args.reward_plot_filename)
    try:
        generate_plots_from_metrics(
            metrics_path=metrics_path,
            success_out=success_plot_path,
            reward_out=reward_plot_path,
            rolling_window=max(10, args.episodes // 10),
            title_prefix="Evaluation (Online)",
            success_time_limit=args.success_time_limit,
        )
        print(f"  Success-rate plot saved -> {success_plot_path}")
        print(f"  Reward plot saved -> {reward_plot_path}")
    except ModuleNotFoundError as exc:
        if exc.name == "matplotlib":
            print("  Plot generation skipped: matplotlib not available in this interpreter.")
            print("  Run plots with: .venv/bin/python results.py")
        else:
            raise

    if args.save_adapted_checkpoint:
        agent.save(args.save_adapted_checkpoint)
        print(f"  Adapted checkpoint saved -> {args.save_adapted_checkpoint}")

    return stats


if __name__ == "__main__":
    evaluate_online(parse_args())