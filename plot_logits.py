"""
Plot maximum logit value per episode for online vs offline Parkinsons evaluation.

Definitions
-----------
"Online" evaluation lets the disease model progress DURING the evaluation
run itself: agent.on_episode_end() is called after every episode
(progressively pruning motivation neurons) and agent.current_episode is
advanced (deepening the parkinsons_rpe transmission/survival decay) —
exactly what evaluate_parkinsons.py does by default.

"Offline" evaluation freezes the agent's impairment level at whatever was
loaded from the checkpoint: on_episode_end() is never called and
current_episode is never advanced, so neuron count and RPE
transmission/survival stay constant for the whole run.

Both agents are loaded from the SAME checkpoint and run for the SAME number
of episodes with the SAME seeds, so any divergence between the two curves
is attributable purely to whether the disease is allowed to progress mid-run.

Logit extraction
-----------------
LowLogitInhibition does not modify the logits it receives — it only reads
them to decide whether to stall. So the pre-softmax logits used internally
by the network are recomputed here by mirroring
ActorCriticNetwork.forward() up through (but not including) the softmax.
The value plotted per episode is the max logit seen at ANY step in that
episode.

Usage
-----
    python plot_max_logit_online_vs_offline.py --checkpoint checkpoints/a2c_rpe_final.pt
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys


def _ensure_runtime() -> None:
    """Re-launch with workspace venv when core dependencies are missing."""
    required = ("numpy", "torch", "matplotlib")
    if all(importlib.util.find_spec(name) is not None for name in required):
        return
    if os.environ.get("HB_PLOT_REEXEC") == "1":
        return
    repo_root = os.path.dirname(os.path.abspath(__file__))
    venv_python = os.path.join(repo_root, ".venv", "bin", "python")
    if os.path.exists(venv_python):
        env = os.environ.copy()
        env["HB_PLOT_REEXEC"] = "1"
        os.execve(venv_python, [venv_python, os.path.abspath(__file__), *sys.argv[1:]], env)


_ensure_runtime()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from parkinsons_a2c_rpe_model import A2CAgent
from pick_and_place_env import PickAndPlaceEnv


def get_motivation_updated_action_logits(network, state_t: torch.Tensor) -> torch.Tensor:
    """Recompute the MOTIVATION-UPDATED pre-softmax action logits for a state.

    In the Parkinsons ActorCriticNetwork.forward(), the motivation gate is
    applied to the shared features (action_hidden = features * motivation_gate,
    then masked by surviving motivation neurons) BEFORE that tensor is passed
    through actor_head[0] to produce the logits. action_logits is then set
    directly to those motivation-modulated logits (LowLogitInhibition only
    reads them to build a stall mask; it does not alter their values). So the
    logits returned here already reflect motivation modulation — they are NOT
    the pre-motivation, features-only logits.
    """
    with torch.no_grad():
        features = network.shared(state_t)
        masked_motivation = (
            network.motivation_head(state_t) * network.motivation_neuron_mask.view(1, -1)
        )
        masked_motivation = masked_motivation * network.motivation_compensation_scale
        motivation_gate = 1.0 + torch.tanh(masked_motivation)

        # This is the motivation-updated hidden representation actually used
        # by the real forward() pass to produce action logits.
        action_hidden = features * motivation_gate
        action_hidden = action_hidden * network.motivation_neuron_mask.view(1, -1)

        motivation_updated_logits = network.actor_head[0](action_hidden)
        motivation_updated_logits, _ = network.low_logit_inhibition(motivation_updated_logits)
    return motivation_updated_logits


def _assert_logits_match_forward(network, state_t: torch.Tensor, atol: float = 1e-5) -> None:
    """Sanity check: recomputed motivation-updated logits must reproduce the
    network's own forward() action_probs when passed through softmax. Guards
    against this script silently drifting from the real model if either is
    edited later.
    """
    with torch.no_grad():
        recomputed_logits = get_motivation_updated_action_logits(network, state_t)
        recomputed_probs = torch.softmax(recomputed_logits, dim=-1)
        forward_probs, _ = network(state_t)
    if not torch.allclose(recomputed_probs, forward_probs, atol=atol):
        raise RuntimeError(
            "Recomputed motivation-updated logits do not reproduce the "
            "network's own forward() output. get_motivation_updated_action_logits() "
            "is out of sync with ActorCriticNetwork.forward() and must be updated."
        )


def build_agent(
    checkpoint: str,
    state_dim: int,
    action_dim: int,
    hidden_dim: int,
    prune_interval_episodes: int,
    prune_neurons_per_interval: int,
    low_logit_threshold: float,
) -> A2CAgent:
    agent = A2CAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        prune_interval_episodes=prune_interval_episodes,
        prune_neurons_per_interval=prune_neurons_per_interval,
        low_logit_threshold=low_logit_threshold,
    )
    agent.load(checkpoint)
    return agent


def run_evaluation(
    agent: A2CAgent,
    env: PickAndPlaceEnv,
    episodes: int,
    seed: int,
    progress_disease: bool,
) -> list[float]:
    """Run `episodes` episodes; return the max logit value seen in each one.

    If progress_disease is True, on_episode_end() and current_episode are
    advanced after every episode ("online"). If False, both are left
    untouched for the whole run ("offline").
    """
    max_logit_per_episode: list[float] = []
    checked_once = False

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        terminated = truncated = False
        episode_max_logit = -float("inf")

        while not (terminated or truncated):
            state_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            if not checked_once:
                _assert_logits_match_forward(agent.network, state_t)
                checked_once = True
            logits = get_motivation_updated_action_logits(agent.network, state_t)
            step_max_logit = float(logits.max().item())
            episode_max_logit = max(episode_max_logit, step_max_logit)

            action, _ = agent.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)

        max_logit_per_episode.append(episode_max_logit)

        if progress_disease:
            agent.on_episode_end()
            agent.current_episode += 1

    return max_logit_per_episode


def make_env(grid_size: int) -> PickAndPlaceEnv:
    env = PickAndPlaceEnv(
        grid_size=grid_size,
        max_steps=200,
        shaping_start=0.0,
        shaping_end=0.0,
    )
    # Evaluation-only reward policy: only PICK/PLACE produce non-zero rewards.
    env.reward_step = 0.0
    env.reward_invalid = 0.0
    env.reward_start = 0.0
    env.shaping_scale = 0.0
    return env


def rolling_mean(values: list[float], window: int) -> np.ndarray:
    arr = np.array(values, dtype=float)
    if window <= 1 or len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot max logit value per episode: Parkinsons online vs offline evaluation."
    )
    parser.add_argument("--checkpoint", type=str, default="checkpoints/a2c_rpe_final.pt")
    parser.add_argument("--grid_size", type=int, default=5)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--prune_interval_episodes", type=int, default=30)
    parser.add_argument("--prune_neurons_per_interval", type=int, default=6)
    parser.add_argument("--low_logit_threshold", type=float, default=-1.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rolling_window", type=int, default=25)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument(
        "--output_filename",
        type=str,
        default="parkinsons_max_logit_online_vs_offline.png",
    )
    parser.add_argument(
        "--metrics_filename",
        type=str,
        default="parkinsons_max_logit_online_vs_offline.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env = make_env(args.grid_size)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    online_agent = build_agent(
        args.checkpoint, state_dim, action_dim, args.hidden_dim,
        args.prune_interval_episodes, args.prune_neurons_per_interval,
        args.low_logit_threshold,
    )
    offline_agent = build_agent(
        args.checkpoint, state_dim, action_dim, args.hidden_dim,
        args.prune_interval_episodes, args.prune_neurons_per_interval,
        args.low_logit_threshold,
    )

    print(f"Running ONLINE evaluation ({args.episodes} episodes, disease progresses during the run)...")
    online_max_logits = run_evaluation(
        online_agent, env, args.episodes, args.seed, progress_disease=True
    )

    print(f"Running OFFLINE evaluation ({args.episodes} episodes, impairment progresses)...")
    offline_max_logits = run_evaluation(
        offline_agent, env, args.episodes, args.seed, progress_disease=True
    )

    os.makedirs(args.results_dir, exist_ok=True)

    # Raw per-episode values, for anyone who wants to re-analyze without rerunning.
    csv_path = os.path.join(args.results_dir, args.metrics_filename)
    with open(csv_path, "w", encoding="utf-8") as fp:
        fp.write("episode,online_max_logit,offline_max_logit\n")
        for ep, (on_v, off_v) in enumerate(zip(online_max_logits, offline_max_logits), start=1):
            fp.write(f"{ep},{on_v},{off_v}\n")
    print(f"Saved raw values -> {csv_path}")

    # Plot.
    fig, ax = plt.subplots(figsize=(10, 6))
    episodes_axis = np.arange(1, args.episodes + 1)

    ax.plot(episodes_axis, online_max_logits, alpha=0.25, color="tab:red")
    ax.plot(episodes_axis, offline_max_logits, alpha=0.25, color="tab:blue")

    if args.rolling_window > 1 and args.episodes >= args.rolling_window:
        online_smooth = rolling_mean(online_max_logits, args.rolling_window)
        offline_smooth = rolling_mean(offline_max_logits, args.rolling_window)
        smooth_axis = episodes_axis[args.rolling_window - 1:]
        ax.plot(
            smooth_axis, online_smooth, color="tab:red", linewidth=2,
            label=f"Online (rolling mean, w={args.rolling_window})",
        )
        ax.plot(
            smooth_axis, offline_smooth, color="tab:blue", linewidth=2,
            label=f"Offline (rolling mean, w={args.rolling_window})",
        )
    else:
        ax.plot([], [], color="tab:red", label="Online")
        ax.plot([], [], color="tab:blue", label="Offline")

    ax.axhline(
        args.low_logit_threshold, color="gray", linestyle="--", linewidth=1,
        label=f"Stall threshold ({args.low_logit_threshold})",
    )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Max logit value")
    ax.set_title("Max Logit Value per Episode: Parkinsons Online vs Offline Evaluation")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(args.results_dir, args.output_filename)
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot -> {out_path}")


if __name__ == "__main__":
    main()