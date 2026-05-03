import pytest
import numpy as np
from src.rl_defender_env import RLDatasetDefenderEnv

def test_env_initialization():
    X = np.random.rand(100, 152)
    y = np.random.randint(0, 2, size=(100,))
    env = RLDatasetDefenderEnv(X, y, max_steps_per_episode=50, shuffle=False)
    
    assert env.observation_space.shape == (152,)
    assert env.action_space.n == 2
    assert env.n_samples == 100

def test_env_step_and_reset():
    X = np.random.rand(10, 10)
    y = np.ones(10)
    env = RLDatasetDefenderEnv(X, y, shuffle=False)
    
    obs, info = env.reset()
    np.testing.assert_allclose(obs, X[0])
    
    obs, reward, terminated, truncated, info = env.step(1)
    np.testing.assert_allclose(obs, X[1])
    assert reward == env.reward_config["tp"]
    assert not terminated
    
    for _ in range(8):
        env.step(0)
    
    obs, reward, terminated, truncated, info = env.step(0)
    assert terminated