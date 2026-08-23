from __future__ import annotations

import numpy as np

from src.agents.smc_learning import SMCLearningAgent
from src.environment import BoatEnv, StateDiscretizer
from src.utils import decayed


def test_paper_decay_schedule():
    assert decayed(0.5, 0.01, 0) == 0.5
    assert np.isclose(decayed(3.0, 0.0001, 1000), 3.0 / 1.1)


def test_environment_bounds_and_action_clip():
    env = BoatEnv(seed=1)
    obs = env.reset(start_y=100)
    assert obs.shape == (5,)
    for _ in range(10):
        obs, reward, done, info = env.step(999.0)
        assert 0 <= obs[0] <= 200
        assert 0 <= obs[1] <= 200
        if done:
            break


def test_terminal_reward_ordering():
    env = BoatEnv()
    assert env.terminal_reward(110.0) == 10.0
    assert env.terminal_reward(100.0) <= 0.0
    assert env.terminal_reward(0.0) == -10.0


def test_discretizer_shape_and_bounds():
    d = StateDiscretizer(10)
    idx = d.encode(np.array([200, 200, 999, -999, 99], dtype=float))
    assert len(idx) == 2
    assert all(0 <= i < 10 for i in idx)


def test_smc_weights_and_ess():
    rng = np.random.default_rng(2)
    agent = SMCLearningAgent(10, rng)
    state = (0, 0, 0, 0, 0)
    ps = agent.get(state)
    assert np.isclose(ps.weights.sum(), 1.0)
    assert np.isclose(agent.effective_sample_size(ps.weights), 10.0)
    idx, _ = agent.select(state)
    agent.update(state, idx, 1.0, None, None, alpha=0.5, gamma=0.99, temperature=1.0)
    assert np.isclose(agent.get(state).weights.sum(), 1.0)
