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
        ↳ [Linear → ReLU] × 2 → Linear  (critic head – outputs V(s))
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        low_logit_threshold: float = -1.2,
        motivation_gain: float = 1.5,
    ):
        super().__init__()
        self.state_dim = int(state_dim)

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
        # Motivation head: maps F(RPE) to a feature gate that modulates
        # actor hidden features before producing a logits adjustment.
        self.motivation_head = nn.Linear(state_dim, hidden_dim)
        self.motivation_to_logits = nn.Linear(hidden_dim, action_dim)
        self.motivation_gain = float(motivation_gain)
        self.low_logit_inhibition = LowLogitInhibition(threshold=low_logit_threshold)
        # Critic: scalar state-value estimate V(s)
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        motivation_signal: torch.Tensor | None = None,
        return_no_action_mask: bool = False,
        return_motivation_prediction: bool = False,
    ):
        """Return action probabilities and value; optionally return no-action mask."""
        features = self.shared(x)

        # Motivation is state-driven; F(RPE) is used as a supervised training target.
        motivation_gate = 1.0 + torch.tanh(self.motivation_head(x))
        motivation_pred = torch.tanh((motivation_gate - 1.0).mean(dim=-1))
        action_hidden = features * motivation_gate
        base_logits = self.actor_head[0](action_hidden)
        motivation_delta = 1.0 + torch.tanh(
            self.motivation_to_logits(action_hidden)
        )

        # Center around 1.0 so motivation is neutral at init (delta≈1.0).
        motivation_scale = 1.0 + self.motivation_gain * (motivation_delta - 1.0)
        motivation_scale = torch.clamp(motivation_scale, min=0.5, max=2.0)
        action_logits = base_logits * motivation_scale

        action_logits, no_action_mask = self.low_logit_inhibition(action_logits)
        action_probs = F.softmax(action_logits, dim=-1)
        value = self.critic_head(features)
        if return_no_action_mask and return_motivation_prediction:
            return action_probs, value, no_action_mask.squeeze(-1), motivation_pred
        if return_no_action_mask:
            return action_probs, value, no_action_mask.squeeze(-1)
        if return_motivation_prediction:
            return action_probs, value, motivation_pred
        return action_probs, value


class LowLogitInhibition(nn.Module):
    """Detect when logits are too weak to trigger a movement decision."""

    def __init__(self, threshold: float = 0.0):
        super().__init__()
        self.threshold = float(threshold)

    def forward(self, action_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        max_logit = action_logits.max(dim=-1, keepdim=True).values
        no_action_mask = max_logit < self.threshold
        return action_logits, no_action_mask


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
        low_logit_threshold: float = -1.2,
        motivation_gain: float = 1.5,
        motivation_loss_coef: float = 0.1,
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.grad_clip_norm = grad_clip_norm
        self.policy_clip_eps = policy_clip_eps
        self.low_logit_threshold = float(low_logit_threshold)
        self.motivation_gain = float(motivation_gain)
        self.motivation_loss_coef = float(motivation_loss_coef)

        self.network = ActorCriticNetwork(
            state_dim,
            action_dim,
            hidden_dim,
            low_logit_threshold=self.low_logit_threshold,
            motivation_gain=self.motivation_gain,
        )
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
            action_probs, _, no_action_mask = self.network(
                state_t,
                return_no_action_mask=True,
            )

        if bool(no_action_mask.item()):
            return self._stall_action_from_state(state), action_probs.squeeze().detach()

        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        return int(action.item()), action_probs.squeeze().detach()

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
        _, values = self.network(states_t)
        values = values.squeeze(-1)

        # Bootstrap next-state values (masked at terminal states)
        with torch.no_grad():
            _, next_values = self.network(next_states_t)
            next_values = next_values.squeeze(-1) * (1.0 - dones_t)

        # ----------------------------------------------------------------
        # One-step TD error (RPE) and GAE advantages
        # ----------------------------------------------------------------
        deltas = rewards_t + self.gamma * next_values - values

        advantages_t = torch.zeros_like(rewards_t)
        gae = torch.tensor(0.0, dtype=torch.float32)
        for t in range(len(rewards) - 1, -1, -1):
            gae = (
                deltas[t]
                + self.gamma * self.gae_lambda * (1.0 - dones_t[t]) * gae
            )
            advantages_t[t] = gae

        returns_t = advantages_t + values.detach()
        rpe = deltas

        # F(RPE): bounded, normalized modulation signal for the motivation head.
        motivation_signal = torch.tanh(
            (rpe - rpe.mean()) / (rpe.std(unbiased=False) + 1e-8)
        ).detach()

        # Keep actor updates on-policy with the same forward path used in rollout.
        action_probs, _, motivation_pred = self.network(
            states_t,
            return_motivation_prediction=True,
        )

        # Train state-driven motivation head against F(RPE).
        motivation_target = motivation_signal
        motivation_loss = F.mse_loss(motivation_pred, motivation_target)

        # Advantage normalization stabilizes policy-gradient updates.
        advantages_norm = (advantages_t - advantages_t.mean()) / (
            advantages_t.std(unbiased=False) + 1e-8
        )

        # Update dopamine with per-transition RPE (preserves variance information).
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
            adv_detached = advantages_norm.detach()
            unclipped = ratios * adv_detached
            clipped = torch.clamp(
                ratios,
                1.0 - self.policy_clip_eps,
                1.0 + self.policy_clip_eps,
            ) * adv_detached
            actor_loss = -torch.min(unclipped, clipped).mean()
            approx_kl = float((old_log_probs - log_probs).mean().item())
        else:
            actor_loss = -(log_probs * advantages_norm.detach()).mean()
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
            + self.motivation_loss_coef * motivation_loss
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
            "motivation_loss": float(motivation_loss.item()),
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
        saved_state = ckpt["network_state_dict"]
        current_state = self.network.state_dict()
        compatible_state = {
            k: v
            for k, v in saved_state.items()
            if k in current_state and current_state[k].shape == v.shape
        }
        load_result = self.network.load_state_dict(
            compatible_state,
            strict=False,
        )

        allowed_missing_prefixes = (
            "motivation_head.",
            "motivation_to_logits.",
        )
        disallowed_missing = [
            key
            for key in load_result.missing_keys
            if not key.startswith(allowed_missing_prefixes)
        ]
        if disallowed_missing or load_result.unexpected_keys:
            raise RuntimeError(
                "Incompatible checkpoint state_dict. "
                f"missing={disallowed_missing}, "
                f"unexpected={load_result.unexpected_keys}"
            )

        try:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except ValueError:
            # Older checkpoints can have optimizer states for fewer parameters.
            pass
        self.episode_rewards = ckpt.get("episode_rewards", [])
        self.episode_lengths = ckpt.get("episode_lengths", [])
