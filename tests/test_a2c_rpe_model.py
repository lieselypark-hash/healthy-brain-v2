"""
Unit tests for the A2C RPE (dopamine) model components:
  - ActorCriticNetwork
  - DopamineModel
  - A2CAgent
"""

import copy
import random

import numpy as np
import pytest
import torch

from a2c_rpe_model import A2CAgent, ActorCriticNetwork, DopamineModel
from parkinsons_a2c_rpe_model import A2CAgent as ParkinsonsA2CAgent


# ---------------------------------------------------------------------------
# ActorCriticNetwork tests
# ---------------------------------------------------------------------------

class TestActorCriticNetwork:
    @pytest.fixture
    def net(self):
        return ActorCriticNetwork(state_dim=9, action_dim=7, hidden_dim=64)

    def test_forward_shapes(self, net):
        batch = torch.randn(4, 9)
        probs, value = net(batch)
        assert probs.shape == (4, 7)
        assert value.shape == (4, 1)

    def test_action_probs_sum_to_one(self, net):
        x = torch.randn(8, 9)
        probs, _ = net(x)
        totals = probs.sum(dim=-1)
        torch.testing.assert_close(totals, torch.ones(8), atol=1e-5, rtol=0)

    def test_action_probs_non_negative(self, net):
        x = torch.randn(8, 9)
        probs, _ = net(x)
        assert (probs >= 0).all()

    def test_single_sample(self, net):
        x = torch.randn(1, 9)
        probs, value = net(x)
        assert probs.shape == (1, 7)
        assert value.shape == (1, 1)

    def test_gradients_flow(self, net):
        x = torch.randn(4, 9)
        probs, value = net(x)
        loss = -probs.log().mean() + value.mean()
        loss.backward()
        for param in net.parameters():
            assert param.grad is not None


# ---------------------------------------------------------------------------
# DopamineModel tests
# ---------------------------------------------------------------------------

class TestDopamineModel:
    def test_initial_tonic_is_zero(self):
        dm = DopamineModel()
        assert dm.tonic_level == 0.0

    def test_update_returns_phasic(self):
        dm = DopamineModel()
        phasic = dm.update(1.0)
        assert isinstance(phasic, float)

    def test_tonic_moves_toward_rpe(self):
        dm = DopamineModel(alpha_tonic=0.5)
        dm.update(10.0)
        assert dm.tonic_level > 0.0

    def test_positive_rpe_positive_phasic_trend(self):
        dm = DopamineModel(alpha_tonic=0.01)
        for _ in range(100):
            dm.update(5.0)
        stats = dm.get_stats()
        # After many positive RPEs the phasic component stays low (signal absorbed into tonic)
        # but the tonic level should be significantly positive
        assert stats["tonic_level"] > 0.0

    def test_history_is_recorded(self):
        dm = DopamineModel()
        for v in [0.1, 0.2, 0.3]:
            dm.update(v)
        assert len(dm.rpe_history) == 3
        assert len(dm.phasic_history) == 3

    def test_window_size_cap(self):
        dm = DopamineModel(window_size=5)
        for i in range(10):
            dm.update(float(i))
        assert len(dm.rpe_history) == 5
        assert len(dm.phasic_history) == 5

    def test_get_stats_empty(self):
        dm = DopamineModel()
        stats = dm.get_stats()
        assert stats["mean_rpe"] == 0.0
        assert stats["tonic_level"] == 0.0

    def test_get_stats_keys(self):
        dm = DopamineModel()
        dm.update(1.0)
        stats = dm.get_stats()
        for key in ("mean_rpe", "std_rpe", "tonic_level", "mean_phasic"):
            assert key in stats


# ---------------------------------------------------------------------------
# A2CAgent tests
# ---------------------------------------------------------------------------

STATE_DIM  = 9
ACTION_DIM = 7


@pytest.fixture
def agent():
    return A2CAgent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=64,
        lr=1e-3,
    )


@pytest.fixture
def sample_batch():
    n = 4
    states      = [np.random.rand(STATE_DIM).astype(np.float32) for _ in range(n)]
    actions     = [np.random.randint(0, ACTION_DIM) for _ in range(n)]
    rewards     = [float(np.random.randn()) for _ in range(n)]
    next_states = [np.random.rand(STATE_DIM).astype(np.float32) for _ in range(n)]
    dones       = [0.0] * (n - 1) + [1.0]
    return states, actions, rewards, next_states, dones


class TestA2CAgentSelectAction:
    def test_action_in_range(self, agent):
        state = np.random.rand(STATE_DIM).astype(np.float32)
        action, probs = agent.select_action(state)
        assert 0 <= action < ACTION_DIM
        assert probs.shape == (ACTION_DIM,)

    def test_deterministic_on_frozen_network(self, agent):
        """With a fixed seed and deterministic network, repeated calls may differ
        (sampling), but the distribution (probs) should be identical."""
        state = np.ones(STATE_DIM, dtype=np.float32)
        _, probs1 = agent.select_action(state)
        _, probs2 = agent.select_action(state)
        torch.testing.assert_close(probs1, probs2)


class TestA2CAgentUpdate:
    def test_update_returns_expected_keys(self, agent, sample_batch):
        info = agent.update(*sample_batch)
        for key in ("actor_loss", "critic_loss", "total_loss", "mean_rpe", "entropy"):
            assert key in info

    def test_update_changes_parameters(self, agent, sample_batch):
        params_before = [p.clone() for p in agent.network.parameters()]
        agent.update(*sample_batch)
        params_after = list(agent.network.parameters())
        # At least one parameter should have changed
        changed = any(
            not torch.equal(b, a) for b, a in zip(params_before, params_after)
        )
        assert changed

    def test_update_records_loss(self, agent, sample_batch):
        agent.update(*sample_batch)
        assert len(agent.training_losses) == 1

    def test_update_triggers_dopamine_update(self, agent, sample_batch):
        _, _, rewards, _, _ = sample_batch
        agent.update(*sample_batch)
        assert len(agent.dopamine.rpe_history) == len(rewards)

    def test_rpe_is_finite(self, agent, sample_batch):
        info = agent.update(*sample_batch)
        assert np.isfinite(info["mean_rpe"])

    def test_entropy_positive(self, agent, sample_batch):
        info = agent.update(*sample_batch)
        assert info["entropy"] >= 0.0

    def test_terminal_state_bootstrapped_to_zero(self):
        """When done=1, the next-state value should be masked to 0."""
        a = A2CAgent(STATE_DIM, ACTION_DIM, hidden_dim=64)
        s  = [np.zeros(STATE_DIM, dtype=np.float32)]
        ac = [0]
        r  = [1.0]
        ns = [np.ones(STATE_DIM, dtype=np.float32)]
        d  = [1.0]  # terminal
        info = a.update(s, ac, r, ns, d)
        assert np.isfinite(info["total_loss"])


class TestA2CAgentPersistence:
    def test_save_and_load(self, agent, tmp_path, sample_batch):
        # Train a little so weights differ from init
        for _ in range(3):
            agent.update(*sample_batch)

        ckpt = str(tmp_path / "test.pt")
        agent.save(ckpt)

        agent2 = A2CAgent(STATE_DIM, ACTION_DIM, hidden_dim=64)
        agent2.load(ckpt)

        # Parameters should be identical after loading
        for p1, p2 in zip(
            agent.network.parameters(), agent2.network.parameters()
        ):
            torch.testing.assert_close(p1, p2)


class TestParkinsonsA2CAgent:
    def test_select_action_returns_unmixed_policy_probs(self):
        torch.manual_seed(0)
        agent = ParkinsonsA2CAgent(
            STATE_DIM,
            ACTION_DIM,
            hidden_dim=64,
            movement_execution_probability=0.0,
            freeze_episode_probability=0.0,
        )

        state = np.ones(STATE_DIM, dtype=np.float32)
        _, sampled_probs = agent.select_action(state)
        with torch.no_grad():
            network_probs, _ = agent.network(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0))
        torch.testing.assert_close(sampled_probs, network_probs.squeeze(0), atol=1e-6, rtol=0.0)

    def test_movement_slowness_blocks_movement_action(self):
        torch.manual_seed(0)
        agent = ParkinsonsA2CAgent(
            STATE_DIM,
            ACTION_DIM,
            hidden_dim=64,
            movement_execution_probability=0.0,
            freeze_episode_probability=0.0,
        )

        with torch.no_grad():
            for p in agent.network.parameters():
                p.zero_()
            agent.network.actor_head[0].bias[0] = 10.0  # force action 0 (UP)

        state = np.zeros(STATE_DIM, dtype=np.float32)
        action, _ = agent.select_action(state)
        assert action == 5  # PICK is invalid when not holding, so movement stalls

    def test_freeze_episode_blocks_multiple_steps(self):
        torch.manual_seed(0)
        random.seed(0)
        agent = ParkinsonsA2CAgent(
            STATE_DIM,
            ACTION_DIM,
            hidden_dim=64,
            movement_execution_probability=1.0,
            freeze_episode_probability=1.0,
            freeze_min_steps=3,
            freeze_max_steps=3,
        )

        with torch.no_grad():
            for p in agent.network.parameters():
                p.zero_()
            agent.network.actor_head[0].bias[0] = 10.0  # force action 0 (UP)

        state = np.zeros(STATE_DIM, dtype=np.float32)
        a1, _ = agent.select_action(state)
        agent.freeze_episode_probability = 0.0  # avoid starting a new freeze episode
        a2, _ = agent.select_action(state)
        a3, _ = agent.select_action(state)
        a4, _ = agent.select_action(state)

        assert (a1, a2, a3) == (5, 5, 5)
        assert a4 == 0

    def test_impaired_rpe_drives_learning_signal(self, sample_batch):
        random.seed(0)
        torch.manual_seed(0)

        normal_agent = A2CAgent(STATE_DIM, ACTION_DIM, hidden_dim=64)
        pd_agent = ParkinsonsA2CAgent(
            STATE_DIM,
            ACTION_DIM,
            hidden_dim=64,
            surviving_fraction=0.0,
            transmission_probability=1.0,
        )
        pd_agent.network.load_state_dict(copy.deepcopy(normal_agent.network.state_dict()))
        pd_agent.optimizer.load_state_dict(copy.deepcopy(normal_agent.optimizer.state_dict()))

        normal_info = normal_agent.update(*sample_batch)
        pd_info = pd_agent.update(*sample_batch)

        assert normal_info["mean_abs_rpe"] > 0.0
        assert pd_info["mean_rpe"] == pytest.approx(0.0)
        assert pd_info["mean_abs_rpe"] == pytest.approx(0.0)
        assert abs(pd_info["actor_loss"]) < abs(normal_info["actor_loss"])
        assert abs(pd_agent.dopamine.get_stats()["mean_rpe"]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Integration: short training smoke-test
# ---------------------------------------------------------------------------

class TestA2CIntegration:
    def test_short_training_does_not_crash(self):
        """Run 10 episodes of training to ensure nothing raises."""
        from pick_and_place_env import PickAndPlaceEnv

        env   = PickAndPlaceEnv(grid_size=4, max_steps=50, seed=0)
        a     = A2CAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            hidden_dim=32,
        )

        for ep in range(10):
            obs, _ = env.reset(seed=ep)
            done = False
            while not done:
                batch_s, batch_a, batch_r, batch_ns, batch_d = [], [], [], [], []
                for _ in range(4):
                    action, _ = a.select_action(obs)
                    next_obs, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    batch_s.append(obs)
                    batch_a.append(action)
                    batch_r.append(reward)
                    batch_ns.append(next_obs)
                    batch_d.append(float(done))
                    obs = next_obs
                    if done:
                        break
                a.update(batch_s, batch_a, batch_r, batch_ns, batch_d)

        assert len(a.training_losses) > 0
        stats = a.dopamine.get_stats()
        assert np.isfinite(stats["tonic_level"])
