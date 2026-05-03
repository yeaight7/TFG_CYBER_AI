import pytest
import numpy as np
from src.rl_defender_env import RLDatasetDefenderEnv

def test_reward_logic_tp_fp_tn_fn():
    X = np.zeros((4, 10))
    y = np.array([0, 1, 0, 1])
    
    config = {"tp": 10.0, "fp": -5.0, "fn": -10.0, "omission": 1.0}
    env = RLDatasetDefenderEnv(X, y, reward_config=config, shuffle=False)
    
    assert env.reward_config["tp"] == 10.0
    assert env.reward_config["fp"] == -5.0
    assert env.reward_config["fn"] == -10.0
    assert env.reward_config["omission"] == 1.0
    
    # 0=PERMIT, 1=BLOCK
    assert env._compute_reward(0, 0) == 1.0   # TN eq
    assert env._compute_reward(0, 1) == -5.0  # FP
    assert env._compute_reward(1, 0) == -10.0 # FN
    assert env._compute_reward(1, 1) == 10.0  # TP

def test_unknown_label_reward():
    X = np.zeros((1, 10))
    y = np.array([0])
    env = RLDatasetDefenderEnv(X, y, shuffle=False)
    
    assert env._compute_reward(99, 1) == env.reward_config["omission"]
    assert env._compute_reward(99, 0) == env.reward_config["fp"]