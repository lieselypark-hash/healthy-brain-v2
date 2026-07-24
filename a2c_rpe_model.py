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
         ↳ Linear                 (critic head  – outputs V(s))
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
        self.critic_head = nn.Linear(hidden_dim, 1)

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
                "std_rpe": 0.0,
                "tonic_level": self.tonic_level,
                "mean_phasic": 0.0,
            }
        rpe_arr = np.array(self._rpe_history)
        ph_arr = np.array(self._phasic_history)
        return {
            "mean_rpe": float(np.mean(rpe_arr)),
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
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        alpha_tonic: float = 0.005,
    ):
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

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
        return int(action.item()), action_probs.squeeze().detach()

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
        # Reward Prediction Error  →  dopamine signal
        #   δ_t = r_t + γ · V(s_{t+1}) − V(s_t)
        # ----------------------------------------------------------------
        rpe = rewards_t + self.gamma * next_values - values.detach()

        # Feed RPE into the dopamine model (updates tonic / phasic signals)
        mean_rpe = float(rpe.mean().item())
        self.dopamine.update(mean_rpe)

        # ----------------------------------------------------------------
        # Actor loss  (policy gradient weighted by advantage = RPE)
        # ----------------------------------------------------------------
        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(actions_t)
        entropy = dist.entropy().mean()

        actor_loss = -(log_probs * rpe.detach()).mean()

        # ----------------------------------------------------------------
        # Critic loss  (minimise squared TD error)
        # ----------------------------------------------------------------
        td_targets = (rewards_t + self.gamma * next_values).detach()
        critic_loss = F.mse_loss(values, td_targets)

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
        nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
        self.optimizer.step()

        self.training_losses.append(float(total_loss.item()))

        return {
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "total_loss": float(total_loss.item()),
            "mean_rpe": mean_rpe,
            "entropy": float(entropy.item()),
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
