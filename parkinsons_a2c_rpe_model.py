"""
A2C with RPE-based Dopamine Model
==================================
Implements an Advantage Actor-Critic (A2C) agent whose learning signal is
grounded in the neuroscience of dopamine.

Neuroscience background
-----------------------
Midbrain dopaminergic neurons encode a *Reward Prediction Error* (RPE):

    δ_t = r_t + γ · V(s_{t+1}) − V(s_t)

* δ_t > 0  →  reward better than expected  →  phasic dopamine *burst*
* δ_t < 0  →  reward worse  than expected  →  phasic dopamine *dip*
* δ_t ≈ 0  →  reward matches prediction    →  baseline (tonic) activity

This is identical to the TD-error used in temporal-difference (TD) learning.
The A2C advantage is therefore a direct model of the dopaminergic teaching
signal: the *actor* (policy) is updated proportionally to δ_t, and the
*critic* (value function) is trained to minimise δ_t².

Classes
-------
ActorCriticNetwork   – shared-trunk neural network with actor and critic heads
DopamineModel        – tracks tonic and phasic dopamine from a stream of RPEs
A2CAgent             – combines the above into a full A2C training agent
"""

from __future__ import annotations

from collections import deque
import random
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------

class ActorCriticNetwork(nn.Module):
    """
    Shared-trunk actor-critic network.

    Architecture
    ------------
    Input → [Linear → ReLU] × 2  (shared feature extractor)
         ↳ Linear → Softmax       (actor  head  – outputs π(a|s))
        ↳ [Linear → ReLU] × 2 → Linear  (critic head – outputs V(s))
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # Actor: action probability distribution
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1),
        )
        # Critic: scalar state-value estimate V(s)
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor):
        """Return (action_probs, value) tensors for a batch of states."""
        features = self.shared(x)
        action_probs = self.actor_head(features)
        value = self.critic_head(features)
        return action_probs, value


# ---------------------------------------------------------------------------
# Dopamine model (RPE tracking)
# ---------------------------------------------------------------------------

class DopamineModel:
    """
    Biologically-inspired dopamine signal derived from RPE.

    The model maintains two timescales that mirror the two roles of
    dopamine in the brain:

    Tonic dopamine
        A slow-moving baseline reflecting average expected reward.
        Updated with a small learning rate (``alpha_tonic``).

    Phasic dopamine
        A fast, transient signal encoding *surprise* relative to the
        tonic baseline.  Only positive deviations are represented as
        phasic bursts; negative deviations correspond to dips below
        baseline.  This is consistent with the asymmetric coding
        observed in dopaminergic neurons.

    Parameters
    ----------
    alpha_tonic : float
        Exponential moving-average coefficient for tonic dopamine update.
    window_size : int
        Length of the rolling history buffers.
    """

    def __init__(self, alpha_tonic: float = 0.005, window_size: int = 1000):
        self.alpha_tonic = alpha_tonic
        self.tonic_level: float = 0.0

        self._rpe_history: deque = deque(maxlen=window_size)
        self._phasic_history: deque = deque(maxlen=window_size)

    def update(self, rpe: float) -> float:
        """
        Receive a new RPE sample and return the phasic dopamine response.

        Parameters
        ----------
        rpe : float
            The current reward prediction error δ_t.

        Returns
        -------
        float
            Phasic dopamine component (≥ 0 for bursts, < 0 for dips).
        """
        self._rpe_history.append(rpe)

        # Tonic update: slow exponential moving average
        self.tonic_level = (
            (1.0 - self.alpha_tonic) * self.tonic_level
            + self.alpha_tonic * rpe
        )

        # Phasic component: deviation from the tonic baseline
        phasic = rpe - self.tonic_level
        self._phasic_history.append(phasic)
        return phasic

    def get_stats(self) -> dict:
        """Return a summary dict of the current dopamine state."""
        if not self._rpe_history:
            return {
                "mean_rpe": 0.0,
                "mean_abs_rpe": 0.0,
                "std_rpe": 0.0,
                "tonic_level": self.tonic_level,
                "mean_phasic": 0.0,
            }
        rpe_arr = np.array(self._rpe_history)
        ph_arr = np.array(self._phasic_history)
        return {
            "mean_rpe": float(np.mean(rpe_arr)),
            "mean_abs_rpe": float(np.mean(np.abs(rpe_arr))),
            "std_rpe": float(np.std(rpe_arr)),
            "tonic_level": self.tonic_level,
            "mean_phasic": float(np.mean(ph_arr)),
        }


    @property
    def rpe_history(self) -> List[float]:
        return list(self._rpe_history)

    @property
    def phasic_history(self) -> List[float]:
        return list(self._phasic_history)


def parkinsons_rpe(
    reward: torch.Tensor,
    gamma: float,
    value: torch.Tensor,
    next_value: torch.Tensor,
    surviving_fraction: float = 0.3,
    transmission_probability: float = 0.3,
) -> torch.Tensor:
    """Return a Parkinson's-modified TD error signal.

    This mirrors a simple neuron-loss plus probabilistic transmission-failure
    model: when transmission occurs, the signal is scaled by
    ``surviving_fraction``; otherwise, it is set to zero.
    """
    delta = reward + gamma * next_value - value
    if random.random() < transmission_probability:
        return surviving_fraction * delta
    return torch.zeros_like(delta)


# ---------------------------------------------------------------------------
# A2C agent
# ---------------------------------------------------------------------------

class A2CAgent:
    """
    Advantage Actor-Critic agent with RPE-modelled dopamine.

    The RPE (δ_t) is computed at every update step and fed to the
    :class:`DopamineModel`, making the dopamine signal an explicit,
    inspectable quantity in the training loop.

    The actor gradient is:

        ∇_θ J ≈ Σ_t  δ_t · ∇_θ log π_θ(a_t | s_t)

    and the critic minimises the mean-squared TD error:

        L_critic = E[ (r_t + γ·V(s_{t+1}) − V(s_t))² ]

    An entropy bonus is added to encourage exploration.

    Parameters
    ----------
    state_dim : int
    action_dim : int
    hidden_dim : int
        Width of the hidden layers.
    lr : float
        Adam optimiser learning rate.
    gamma : float
        Discount factor.
    entropy_coef : float
        Weight on the entropy bonus (encourages exploration).
    value_coef : float
        Weight on the critic loss relative to the actor loss.
    alpha_tonic : float
        Passed through to :class:`DopamineModel`.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        alpha_tonic: float = 0.005,
        grad_clip_norm: float = 0.5,
        policy_clip_eps: float = 0.2,
        surviving_fraction: float = 0.3,
        transmission_probability: float = 0.3,
        movement_execution_probability: float = 0.55,
        freeze_episode_probability: float = 0.12,
        freeze_min_steps: int = 4,
        freeze_max_steps: int = 12,
    ):

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.grad_clip_norm = grad_clip_norm
        self.policy_clip_eps = policy_clip_eps
        self.surviving_fraction = surviving_fraction
        self.transmission_probability = transmission_probability
        self.movement_execution_probability = float(
            np.clip(movement_execution_probability, 0.0, 1.0)
        )
        self.freeze_episode_probability = float(
            np.clip(freeze_episode_probability, 0.0, 1.0)
        )
        self.freeze_min_steps = max(1, int(freeze_min_steps))
        self.freeze_max_steps = max(self.freeze_min_steps, int(freeze_max_steps))
        self._freeze_steps_remaining = 0

        self.network = ActorCriticNetwork(state_dim, action_dim, hidden_dim)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)

        # Dopamine model – tracks tonic and phasic RPE signals
        self.dopamine = DopamineModel(alpha_tonic=alpha_tonic)

        # Book-keeping
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.training_losses: List[float] = []

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray):
        """
        Sample an action from the current policy π(·|s).

        Returns
        -------
        action : int
        action_probs : torch.Tensor  (detached)
        """
        state_t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action_probs, _ = self.network(state_t)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        action_id = int(action.item())

        # Motor impairment layer: keep decision policy intact, but stochastically
        # block execution of movement and START actions to model slowness and
        # freeze episodes.
        if action_id in (0, 1, 2, 3, 6):
            if self._freeze_steps_remaining > 0:
                self._freeze_steps_remaining -= 1
                action_id = self._stall_action_from_state(state)
            else:
                if random.random() < self.freeze_episode_probability:
                    self._freeze_steps_remaining = random.randint(
                        self.freeze_min_steps,
                        self.freeze_max_steps,
                    ) - 1
                    action_id = self._stall_action_from_state(state)
                elif random.random() > self.movement_execution_probability:
                    action_id = self._stall_action_from_state(state)

        return action_id, action_probs.squeeze().detach()

    def _stall_action_from_state(self, state: np.ndarray) -> int:
        """Choose a mostly invalid non-movement action to simulate no movement."""
        holding = bool(float(state[4]) >= 0.5)
        # PLACE is invalid when not holding; PICK is invalid when already holding.
        return 4 if holding else 5

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def update(
        self,
        states: List[np.ndarray],
        actions: List[int],
        rewards: List[float],
        next_states: List[np.ndarray],
        dones: List[float],
        old_action_probs: List[float] | None = None,
    ) -> dict:
        """
        Perform one A2C parameter update on a collected mini-batch.

        The RPE (δ_t = r + γ·V(s') − V(s)) is computed for every
        transition, fed to the :class:`DopamineModel`, and used as the
        advantage estimate for the actor gradient.

        Parameters
        ----------
        states, actions, rewards, next_states, dones :
            Parallel lists of transition data.

        Returns
        -------
        dict
            Keys: actor_loss, critic_loss, total_loss, mean_rpe, entropy.
        """
        states_t = torch.as_tensor(np.array(states), dtype=torch.float32)
        actions_t = torch.as_tensor(actions, dtype=torch.long)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32)
        next_states_t = torch.as_tensor(np.array(next_states), dtype=torch.float32)
        dones_t = torch.as_tensor(dones, dtype=torch.float32)

        # Forward pass
        action_probs, values = self.network(states_t)
        values = values.squeeze(-1)

        # Bootstrap next-state values (masked at terminal states)
        with torch.no_grad():
            _, next_values = self.network(next_states_t)
            next_values = next_values.squeeze(-1) * (1.0 - dones_t)

        # ----------------------------------------------------------------
        # One-step TD error (RPE) and GAE advantages
        # ----------------------------------------------------------------
        deltas = rewards_t + self.gamma * next_values - values

        pd_deltas = torch.stack(
            [
                parkinsons_rpe(
                    reward=rewards_t[t],
                    gamma=self.gamma,
                    value=values[t],
                    next_value=next_values[t],
                    surviving_fraction=self.surviving_fraction,
                    transmission_probability=self.transmission_probability,
                )
                for t in range(len(rewards))
            ]
        )

        advantages_t = torch.zeros_like(rewards_t)
        gae = torch.tensor(0.0, dtype=torch.float32)
        for t in range(len(rewards) - 1, -1, -1):
            gae = (
                pd_deltas[t]
                + self.gamma * self.gae_lambda * (1.0 - dones_t[t]) * gae
            )
            advantages_t[t] = gae

        returns_t = advantages_t + values.detach()
        rpe = pd_deltas

        # Update dopamine with the same impaired RPE that drives learning.
        for rpe_value in rpe.detach().cpu().tolist():
            self.dopamine.update(float(rpe_value))

        mean_rpe = float(rpe.detach().mean().item())
        mean_abs_rpe = float(rpe.detach().abs().mean().item())

        # ----------------------------------------------------------------
        # Actor loss  (policy gradient weighted by advantage = RPE)
        # ----------------------------------------------------------------
        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(actions_t)
        entropy = dist.entropy().mean()

        if old_action_probs is not None:
            old_probs_t = torch.as_tensor(old_action_probs, dtype=torch.float32)
            old_log_probs = torch.log(old_probs_t.clamp_min(1e-8))
            ratios = torch.exp(log_probs - old_log_probs)
            adv_detached = advantages_t.detach()
            unclipped = ratios * adv_detached
            clipped = torch.clamp(
                ratios,
                1.0 - self.policy_clip_eps,
                1.0 + self.policy_clip_eps,
            ) * adv_detached
            actor_loss = -torch.min(unclipped, clipped).mean()
            approx_kl = float((old_log_probs - log_probs).mean().item())
        else:
            actor_loss = -(log_probs * advantages_t.detach()).mean()
            approx_kl = 0.0

        # ----------------------------------------------------------------
        # Critic loss  (minimise squared TD error)
        # ----------------------------------------------------------------
        critic_loss = F.mse_loss(values, returns_t.detach())

        # ----------------------------------------------------------------
        # Combined loss
        # ----------------------------------------------------------------
        total_loss = (
            actor_loss
            + self.value_coef * critic_loss
            - self.entropy_coef * entropy
        )

        self.optimizer.zero_grad()
        total_loss.backward()
        # Gradient clipping for training stability
        nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=self.grad_clip_norm)
        self.optimizer.step()

        self.training_losses.append(float(total_loss.item()))

        return {
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "total_loss": float(total_loss.item()),
            "mean_rpe": mean_rpe,
            "mean_abs_rpe": mean_abs_rpe,
            "entropy": float(entropy.item()),
            "approx_kl": approx_kl,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save network weights, optimiser state, and training history."""
        torch.save(
            {
                "network_state_dict": self.network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "episode_rewards": self.episode_rewards,
                "episode_lengths": self.episode_lengths,
                "dopamine_stats": self.dopamine.get_stats(),
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load a previously saved checkpoint."""
        ckpt = torch.load(path, weights_only=False)
        self.network.load_state_dict(ckpt["network_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.episode_rewards = ckpt.get("episode_rewards", [])
        self.episode_lengths = ckpt.get("episode_lengths", [])
