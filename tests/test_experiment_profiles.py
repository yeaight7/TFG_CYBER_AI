import pytest

from src.experiment_profiles import (
    MAIN_V1_PROFILE,
    MAIN_V1_PROFILE_HASH,
    get_experiment_profile,
)


EXPECTED_MAIN_V1_CONTENT = {
    "policy": "MlpPolicy",
    "policy_kwargs": {
        "net_arch": [1024, 1024, 512],
        "n_quantiles": 200,
    },
    "learning_rate": 5e-5,
    "buffer_size": 1_000_000,
    "learning_starts": 50_000,
    "batch_size": 2_048,
    "gamma": 0.0,
    "tau": 1.0,
    "train_freq": 100,
    "gradient_steps": 20,
    "target_update_interval": 10_000,
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.02,
    "exploration_fraction": 0.10,
    "max_grad_norm": 10.0,
    "reward_config": {
        "tp": 1.5,
        "fp": -2.0,
        "fn": -5.0,
        "omission": 0.0,
    },
}


def test_main_v1_profile_values_and_hash_are_frozen():
    assert MAIN_V1_PROFILE.profile_id == "main-v1"
    assert MAIN_V1_PROFILE.to_dict() == EXPECTED_MAIN_V1_CONTENT
    assert MAIN_V1_PROFILE.content_hash == MAIN_V1_PROFILE_HASH
    assert MAIN_V1_PROFILE_HASH == "17bbeb3f8020f7a1f8860e70b9fbf65b495f71d3dc40e1e30f24dfa86299a19a"


def test_profile_returns_defensive_copies_and_resolves_by_id():
    content = MAIN_V1_PROFILE.to_dict()
    content["policy_kwargs"]["net_arch"][0] = 1
    content["reward_config"]["tp"] = 0.0

    assert MAIN_V1_PROFILE.to_dict() == EXPECTED_MAIN_V1_CONTENT
    assert get_experiment_profile("main-v1") is MAIN_V1_PROFILE

    with pytest.raises(ValueError, match="Unknown experiment profile"):
        get_experiment_profile("missing")
