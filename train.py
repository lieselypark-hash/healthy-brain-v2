"""
Training script for the A2C + RPE (Dopamine) Pick-and-Place agent.

Usage
-----
    python train.py                           # default settings
    python train.py --n_episodes 10000 --lr 3e-4 --grid_size 6
    python train.py --n_episodes 500 --no_save   # quick smoke-test

Key parameters
--------------
--n_episodes    Total number of training episodes.
--n_steps       Number of environment steps before each network update.
--grid_size     Side-length of the square pick-and-place grid.
--hidden_dim    Width of the shared hidden layers.
--lr            Adam learning rate.
--gamma         Discount factor.
--entropy_coef  Entropy bonus weight (exploration).
--value_coef    Critic loss weight.
--log_interval  How many episodes between console logs.
--save_interval How many episodes between checkpoint saves.
--save_dir      Directory for checkpoint files.
--no_save       Disable saving checkpoints.
--seed          Random seed for reproducibility.
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import os
import sys


def _ensure_training_runtime() -> None:
    """Re-launch with workspace venv when core dependencies are missing."""
    required = ("numpy", "torch")
    if all(importlib.util.find_spec(name) is not None for name in required):
        return

    if os.environ.get("HB_TRAIN_REEXEC") == "1":
        return

    repo_root = os.path.dirname(os.path.abspath(__file__))
    venv_python = os.path.join(repo_root, ".venv", "bin", "python")
    if os.path.exists(venv_python):
        env = os.environ.copy()
        env["HB_TRAIN_REEXEC"] = "1"
        os.execve(venv_python, [venv_python, os.path.abspath(__file__), *sys.argv[1:]], env)


_ensure_training_runtime()

import numpy as np
import torch

from a2c_rpe_model import A2CAgent as NormalA2CAgent
from parkinsons_a2c_rpe_model import A2CAgent as ParkinsonsA2CAgent
from pick_and_place_env import PickAndPlaceEnv
from results import generate_plots_from_metrics


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train A2C with RPE (dopamine) on the Pick-and-Place task."
    )
    parser.add_argument("--n_episodes",  type=int,   default=10000)
    parser.add_argument("--n_steps",     type=int,   default=16,
                        help="Number of steps per A2C update.")
    parser.add_argument("--grid_size",   type=int,   default=5)
    parser.add_argument("--agent_variant", type=str, default="normal",
                        choices=(
                            "normal",
                            "normal_no_shaping",
                            "parkinsons",
                            "parkinsons_no_shaping",
                            "parkinsons_ldopa",
                            "parkinsons_ldopa_no_shaping",
                        ),
                        help="Which agent dynamics and shaping regime to train.")
    parser.add_argument("--hidden_dim",  type=int,   default=128)
    parser.add_argument("--lr",          type=float, default=2e-4)
    parser.add_argument("--gamma",       type=float, default=0.99)
    parser.add_argument("--gae_lambda",  type=float, default=0.95)
    parser.add_argument("--entropy_coef",type=float, default=0.08,
                        help="Initial entropy bonus weight for exploration.")
    parser.add_argument("--entropy_coef_final", type=float, default=0.01,
                        help="Final entropy bonus weight after decay.")
    parser.add_argument("--schedule_window", type=int, default=200,
                        help="Rolling window size used to estimate current success.")
    parser.add_argument("--schedule_target_success", type=float, default=0.95,
                        help="Success threshold where curriculum/entropy reach final values.")
    parser.add_argument("--schedule_warmup_episodes", type=int, default=100,
                        help="Episodes before adaptive schedule starts updating.")
    parser.add_argument("--value_coef",  type=float, default=0.5)
    parser.add_argument("--grad_clip_norm", type=float, default=0.5)
    parser.add_argument("--policy_clip_eps", type=float, default=0.1,
                        help="Clipping epsilon for ratio-based policy update.")
    parser.add_argument("--min_lr_ratio", type=float, default=0.3,
                        help="Lower bound for LR annealing as a fraction of base LR.")
    parser.add_argument("--alpha_tonic", type=float, default=0.005,
                        help="Tonic dopamine EMA coefficient.")
    parser.add_argument("--surviving_fraction", type=float, default=1.0,
                        help="Initial Parkinson's dopamine signal scale when transmitted; "
                             "decays toward --min_surviving_fraction as motivation neurons are pruned.")
    parser.add_argument("--transmission_probability", type=float, default=1.0,
                        help="Initial probability that a Parkinson's dopamine signal is transmitted; "
                             "decays toward --min_transmission_probability as motivation neurons are pruned.")
    parser.add_argument("--min_surviving_fraction", type=float, default=0.3,
                        help="Floor for the Parkinson's dopamine signal scale.")
    parser.add_argument("--min_transmission_probability", type=float, default=0.3,
                        help="Floor for the Parkinson's dopamine transmission probability.")
    parser.add_argument("--log_interval",   type=int, default=100)
    parser.add_argument("--save_interval",  type=int, default=500)
    parser.add_argument("--save_dir",       type=str, default="checkpoints")
    parser.add_argument("--results_dir",    type=str, default="results",
                        help="Directory to store training metrics and plots.")
    parser.add_argument("--metrics_filename", type=str, default="training_metrics.csv",
                        help="CSV filename for saved episode metrics.")
    parser.add_argument("--success_plot_filename", type=str, default="success_rate.png",
                        help="Filename for training success-rate plot.")
    parser.add_argument("--reward_plot_filename", type=str, default="reward.png",
                        help="Filename for training reward plot.")
    parser.add_argument("--best_window",    type=int, default=200,
                        help="Rolling window length for selecting best policy.")
    parser.add_argument("--best_start_episode", type=int, default=200,
                        help="Start selecting best policy after this many episodes.")
    parser.add_argument("--no_save",        action="store_true")
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--success_time_limit", type=int, default=75,
                        help="Count success only when placement occurs within this many steps.")
    return parser.parse_args()


def _is_timed_success(info: dict, episode_length: int, success_time_limit: int) -> bool:
    """Return True only when the task is completed within the step limit."""
    return bool(info.get("object_placed", False)) and int(episode_length) <= int(success_time_limit)


def save_training_metrics(path: str, rows: list[dict]) -> None:
    """Persist per-episode training metrics to CSV."""
    if not rows:
        return

    fieldnames = [
        "episode",
        "episode_reward",
        "episode_length",
        "completed_task",
        "success",
        "started_task",
        "cumulative_success_rate",
        "rolling_success_rate",
        "cumulative_start_rate",
        "rolling_start_rate",
        "tonic_dopamine",
        "mean_rpe",
        "mean_abs_rpe",
        "entropy_coef",
        "learning_rate",
        "schedule_progress",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> tuple[A2CAgent, list, list]:
    """
    Train the A2C agent and return (agent, episode_rewards, episode_lengths).

    The training loop follows the standard n-step A2C pattern:
      1. Collect ``n_steps`` transitions (or until episode end).
      2. Compute RPE for each transition.
      3. Update actor and critic.
      4. Repeat until episode terminates.
    """
    np.random.seed(args.seed)

    no_reward_shaping = args.agent_variant.endswith("_no_shaping")
    env_kwargs = {
        "grid_size": args.grid_size,
        "max_steps": 200,
        "seed": args.seed,
        "shaping_start": 1.0,
        "shaping_end": 0.0,
    }
    if no_reward_shaping:
        env_kwargs.update({"shaping_start": 0.0, "shaping_end": 0.0})
    env = PickAndPlaceEnv(**env_kwargs)
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent_cls = (
        NormalA2CAgent
        if args.agent_variant.startswith("normal")
        else ParkinsonsA2CAgent
    )
    agent_kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "hidden_dim": args.hidden_dim,
        "lr": args.lr,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "entropy_coef": args.entropy_coef,
        "value_coef": args.value_coef,
        "alpha_tonic": args.alpha_tonic,
        "grad_clip_norm": args.grad_clip_norm,
        "policy_clip_eps": args.policy_clip_eps,
    }
    if args.agent_variant.startswith("parkinsons"):
        agent_kwargs.update(
            {
                "surviving_fraction": args.surviving_fraction,
                "transmission_probability": args.transmission_probability,
                "min_surviving_fraction": args.min_surviving_fraction,
                "min_transmission_probability": args.min_transmission_probability,
                "ldopa_compensation": "ldopa" in args.agent_variant,
            }
        )
    agent = agent_cls(**agent_kwargs)

    if not args.no_save:
        os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 60)
    print("  A2C + RPE Dopamine Model – Pick-and-Place Task")
    print("=" * 60)
    print(f"  Grid size  : {args.grid_size}×{args.grid_size}")
    print(f"  Agent      : {args.agent_variant}")
    shaping_mode = "disabled" if no_reward_shaping else "curriculum"
    print(f"  Reward shaping: {shaping_mode}")
    print(f"  State dim  : {state_dim}   Action dim: {action_dim}")
    print(f"  Hidden dim : {args.hidden_dim}")
    print(f"  LR={args.lr}  γ={args.gamma}  λ={args.gae_lambda}  n_steps={args.n_steps}")
    print(f"  Success criterion: placement within {args.success_time_limit} steps")
    print(
        f"  Entropy: {args.entropy_coef} → {args.entropy_coef_final} "
        f"(adaptive target success {args.schedule_target_success:.2f})"
    )
    print("=" * 60)

    episode_rewards: list[float] = []
    episode_lengths: list[int]   = []
    episode_successes: list[int] = []
    episode_starts: list[int] = []
    success_count = 0
    start_count = 0
    best_success = -1.0
    best_episode = -1
    best_state_dict = None
    restore_events = 0
    last_restore_episode = -10**9
    schedule_progress = 0.0
    rolling_success = 0.0
    base_lr = args.lr
    metrics_rows: list[dict] = []

    # Stability guards to avoid late-stage collapse under maximum curriculum.
    schedule_ramp_step = 0.01
    schedule_backoff_step = 0.02
    restore_drop_margin = 0.12
    restore_backoff = 0.20
    min_best_for_restore = 0.50
    restore_cooldown = max(args.best_window, args.log_interval)

    for episode in range(args.n_episodes):
        if episode + 1 > args.schedule_warmup_episodes and episode_successes:
            window = min(args.schedule_window, len(episode_successes))
            rolling_success = float(np.mean(episode_successes[-window:]))
            target = max(args.schedule_target_success, 1e-6)
            perf_progress = float(np.clip(rolling_success / target, 0.0, 1.0))
            # Allow controlled backoff when performance drops.
            if perf_progress >= schedule_progress:
                schedule_progress = min(
                    1.0,
                    schedule_progress + min(schedule_ramp_step, perf_progress - schedule_progress),
                )
            else:
                schedule_progress = max(
                    perf_progress,
                    schedule_progress - min(schedule_backoff_step, schedule_progress - perf_progress),
                )

        agent.entropy_coef = (
            args.entropy_coef
            + (args.entropy_coef_final - args.entropy_coef) * schedule_progress
        )
        # Use a monotonic curriculum for shaping so it steadily decays to 0 by
        # the end of training, independent of adaptive schedule backoffs.
        shaping_progress = (episode + 1) / max(args.n_episodes, 1)
        env.set_curriculum(shaping_progress)
        current_lr = base_lr * (
            1.0 - (1.0 - args.min_lr_ratio) * schedule_progress
        )
        for param_group in agent.optimizer.param_groups:
            param_group["lr"] = current_lr

        obs, _ = env.reset(seed=args.seed + episode)
        ep_reward = 0.0
        ep_length = 0
        terminated = truncated = False
        last_info: dict = {}

        # Collect experience and update in n-step chunks
        while not (terminated or truncated):
            batch_s, batch_a, batch_r, batch_ns, batch_d, batch_old_p = [], [], [], [], [], []

            for _ in range(args.n_steps):
                action, action_probs = agent.select_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)

                batch_s.append(obs)
                batch_a.append(action)
                batch_r.append(reward)
                batch_ns.append(next_obs)
                batch_d.append(float(terminated or truncated))
                batch_old_p.append(float(action_probs[action].item()))

                ep_reward += reward
                ep_length += 1
                obs = next_obs
                last_info = info

                if terminated or truncated:
                    break

            agent.update(batch_s, batch_a, batch_r, batch_ns, batch_d, batch_old_p)

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)
        agent.episode_rewards.append(ep_reward)
        agent.episode_lengths.append(ep_length)

        completed_task = bool(last_info.get("object_placed", False))
        timed_success = _is_timed_success(last_info, ep_length, args.success_time_limit)

        if timed_success:
            success_count += 1
            episode_successes.append(1)
        else:
            episode_successes.append(0)

        started_task = int(last_info.get("task_started", False))
        episode_starts.append(started_task)
        start_count += started_task

        if hasattr(agent, "on_episode_end"):
            agent.on_episode_end()

        if episode + 1 >= args.best_start_episode:
            w = min(args.best_window, len(episode_successes))
            rolling_success = float(np.mean(episode_successes[-w:]))
            if rolling_success > best_success:
                best_success = rolling_success
                best_episode = episode + 1
                best_state_dict = copy.deepcopy(agent.network.state_dict())

            # If rolling performance collapses far below the best seen level,
            # restore known-good weights and ease curriculum difficulty.
            if (
                best_state_dict is not None
                and best_success >= min_best_for_restore
                and (episode + 1 - last_restore_episode) >= restore_cooldown
                and rolling_success < (best_success - restore_drop_margin)
            ):
                agent.network.load_state_dict(best_state_dict)
                schedule_progress = max(0.0, schedule_progress - restore_backoff)
                last_restore_episode = episode + 1
                restore_events += 1
                print(
                    f"  [stability restore @ ep {episode+1}: "
                    f"rolling={rolling_success:.3f}, best={best_success:.3f}, "
                    f"schedule={schedule_progress:.2f}]"
                )

        da = agent.dopamine.get_stats()
        cumulative_success_rate = success_count / (episode + 1)
        cumulative_start_rate = start_count / (episode + 1)
        rolling_window = min(args.schedule_window, len(episode_starts))
        rolling_start_rate = float(np.mean(episode_starts[-rolling_window:]))
        metrics_rows.append(
            {
                "episode": episode + 1,
                "episode_reward": float(ep_reward),
                "episode_length": int(ep_length),
                "completed_task": int(completed_task),
                "success": int(episode_successes[-1]),
            "started_task": started_task,
                "cumulative_success_rate": float(cumulative_success_rate),
                "rolling_success_rate": float(rolling_success),
            "cumulative_start_rate": float(cumulative_start_rate),
            "rolling_start_rate": float(rolling_start_rate),
                "tonic_dopamine": float(da["tonic_level"]),
                "mean_rpe": float(da["mean_rpe"]),
                "mean_abs_rpe": float(da["mean_abs_rpe"]),
                "entropy_coef": float(agent.entropy_coef),
                "learning_rate": float(current_lr),
                "schedule_progress": float(schedule_progress),
            }
        )

        # Logging
        if (episode + 1) % args.log_interval == 0:
            window = slice(max(0, episode + 1 - args.log_interval), episode + 1)
            avg_r  = np.mean(episode_rewards[window])
            avg_l  = np.mean(episode_lengths[window])
            s_rate = success_count / (episode + 1)
            da = agent.dopamine.get_stats()

            print(
                f"Ep {episode+1:>5}/{args.n_episodes}  "
                f"AvgReward: {avg_r:+7.3f}  "
                f"AvgLen: {avg_l:6.1f}  "
                f"TimedSuccess: {s_rate:.3f}  "
                f"StartRate: {cumulative_start_rate:.3f}  "
                f"RollSuccess: {rolling_success:.3f}  "
                f"LR: {current_lr:.6f}  "
                f"RPE(mean): {da['mean_rpe']:+.4f}  "
                f"RPE(|mean|): {da['mean_abs_rpe']:+.4f}  "
                f"Tonic DA: {da['tonic_level']:+.4f}"
            )

        # Checkpoint
        if not args.no_save and (episode + 1) % args.save_interval == 0:
            ckpt_path = os.path.join(
                args.save_dir, f"a2c_rpe_ep{episode+1}.pt"
            )
            agent.save(ckpt_path)
            print(f"  [checkpoint saved → {ckpt_path}]")

    print()
    print("=" * 60)
    print(f"  Training complete. Episodes: {args.n_episodes}")
    print(f"  Overall timed success rate: {success_count / args.n_episodes:.3f}")
    da = agent.dopamine.get_stats()
    print(f"  Final tonic dopamine level: {da['tonic_level']:.4f}")
    print(f"  Start rate: {start_count / args.n_episodes:.3f}")
    print(f"  Mid-training stability restores: {restore_events}")
    if best_state_dict is not None:
        agent.network.load_state_dict(best_state_dict)
        print(
            f"  Restored best policy from episode {best_episode} "
            f"(rolling success={best_success:.3f})"
        )
    print("=" * 60)

    if not args.no_save:
        final_path = os.path.join(args.save_dir, "a2c_rpe_final.pt")
        agent.save(final_path)
        print(f"  Final model saved → {final_path}")

    os.makedirs(args.results_dir, exist_ok=True)
    metrics_path = os.path.join(args.results_dir, args.metrics_filename)
    save_training_metrics(metrics_path, metrics_rows)
    print(f"  Training metrics saved → {metrics_path}")

    success_plot_path = os.path.join(args.results_dir, args.success_plot_filename)
    reward_plot_path = os.path.join(args.results_dir, args.reward_plot_filename)
    try:
        generate_plots_from_metrics(
            metrics_path=metrics_path,
            success_out=success_plot_path,
            reward_out=reward_plot_path,
            rolling_window=max(10, args.log_interval),
            title_prefix="Training",
        )
        print(f"  Success-rate plot saved → {success_plot_path}")
        print(f"  Reward plot saved → {reward_plot_path}")
    except ModuleNotFoundError as exc:
        if exc.name == "matplotlib":
            print("  Plot generation skipped: matplotlib not available in this interpreter.")
            print("  Run plots with: .venv/bin/python results.py")
        else:
            raise

    return agent, episode_rewards, episode_lengths


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    train(args)
