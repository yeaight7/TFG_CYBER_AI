import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from train_rl_defender import (  # noqa: E402
    parse_args,
    resolve_checkpoint_freq,
    resolve_seeds,
    resolve_total_timesteps,
    resolve_training_hyperparams,
    seed_model_rngs,
)


def _args(timesteps=None, training_profile="default"):
    return argparse.Namespace(
        timesteps=timesteps,
        training_profile=training_profile,
    )


def test_default_profile_preserves_fast_hyperparams():
    total_timesteps = resolve_total_timesteps(_args(), is_fast=True)
    hyperparams = resolve_training_hyperparams(
        training_profile="default",
        is_fast=True,
        total_timesteps=total_timesteps,
    )

    assert total_timesteps == 25_000
    assert hyperparams["policy"] == "MlpPolicy"
    assert hyperparams["policy_kwargs"] == {
        "net_arch": [512, 256],
        "n_quantiles": 200,
    }
    assert hyperparams["learning_rate"] == 1e-4
    assert hyperparams["buffer_size"] == 25_000
    assert hyperparams["learning_starts"] == 100
    assert hyperparams["batch_size"] == 512
    assert hyperparams["gamma"] == 0.0
    assert hyperparams["gradient_steps"] == 10
    assert hyperparams["train_freq"] == 50
    assert hyperparams["target_update_interval"] == 1_000
    assert hyperparams["exploration_fraction"] == 0.005
    assert hyperparams["exploration_final_eps"] == 0.01
    assert hyperparams["max_grad_norm"] is None


def test_default_profile_preserves_full_hyperparams():
    total_timesteps = resolve_total_timesteps(_args(), is_fast=False)
    hyperparams = resolve_training_hyperparams(
        training_profile="default",
        is_fast=False,
        total_timesteps=total_timesteps,
    )

    assert total_timesteps == 100_000
    assert hyperparams["buffer_size"] == 100_000
    assert hyperparams["batch_size"] == 2048
    assert hyperparams["gamma"] == 0.0
    assert hyperparams["gradient_steps"] == 20
    assert hyperparams["train_freq"] == 100
    assert hyperparams["target_update_interval"] == 10_000


def test_main_experiment_profile_resolves_fixed_config():
    total_timesteps = resolve_total_timesteps(
        _args(training_profile="main-experiment"),
        is_fast=False,
    )
    hyperparams = resolve_training_hyperparams(
        training_profile="main-experiment",
        is_fast=False,
        total_timesteps=total_timesteps,
    )

    assert total_timesteps == 3_000_000
    assert hyperparams == {
        "policy": "MlpPolicy",
        "policy_kwargs": {
            "net_arch": [1024, 1024, 512],
            "n_quantiles": 200,
        },
        "learning_rate": 5e-5,
        "buffer_size": 1_000_000,
        "learning_starts": 50_000,
        "batch_size": 2048,
        "gamma": 0.0,
        "tau": 1.0,
        "train_freq": 100,
        "gradient_steps": 20,
        "target_update_interval": 10_000,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.02,
        "exploration_fraction": 0.10,
        "max_grad_norm": 10.0,
    }


def test_explicit_timesteps_override_main_profile_default():
    total_timesteps = resolve_total_timesteps(
        _args(timesteps=1_000, training_profile="main-experiment"),
        is_fast=True,
    )

    assert total_timesteps == 1_000


def test_checkpoint_frequency_defaults_and_overrides():
    assert resolve_checkpoint_freq("default", None) == 0
    assert resolve_checkpoint_freq("main-experiment", None) == 250_000
    assert resolve_checkpoint_freq("main-experiment", 0) == 0
    assert resolve_checkpoint_freq("default", 100_000) == 100_000


def test_seed_defaults_and_explicit_separation():
    defaults = resolve_seeds(seed=None, split_seed=None, model_seed=None)
    assert defaults.split_seed == 42
    assert defaults.model_seed == 42
    assert defaults.legacy_seed_used is False

    explicit = resolve_seeds(seed=None, split_seed=7, model_seed=99)
    assert explicit.split_seed == 7
    assert explicit.model_seed == 99
    assert explicit.legacy_seed_used is False


def test_legacy_seed_sets_both_seeds_only_when_explicit_seeds_are_absent():
    legacy = resolve_seeds(seed=17, split_seed=None, model_seed=None)
    assert legacy.split_seed == 17
    assert legacy.model_seed == 17
    assert legacy.legacy_seed_used is True

    with pytest.raises(ValueError, match="--seed cannot be combined"):
        resolve_seeds(seed=17, split_seed=17, model_seed=None)
    with pytest.raises(ValueError, match="--seed cannot be combined"):
        resolve_seeds(seed=17, split_seed=None, model_seed=17)

    parsed = parse_args(["--seed", "17"])
    assert parsed.split_seed == 17
    assert parsed.model_seed == 17
    assert parsed.legacy_seed_used is True

    parsed_explicit = parse_args(["--split-seed", "7", "--model-seed", "99"])
    assert parsed_explicit.split_seed == 7
    assert parsed_explicit.model_seed == 99
    assert parsed_explicit.legacy_seed_used is False


def test_cli_rejects_mixed_legacy_and_explicit_seed_flags():
    with pytest.raises(SystemExit) as exc_info:
        parse_args(["--seed", "17", "--model-seed", "18"])

    assert exc_info.value.code == 2


def test_model_seed_deterministically_controls_python_numpy_and_torch_rngs():
    seed_model_rngs(73)
    first = (random.random(), np.random.random(), torch.rand(1).item())

    seed_model_rngs(73)
    second = (random.random(), np.random.random(), torch.rand(1).item())

    seed_model_rngs(74)
    third = (random.random(), np.random.random(), torch.rand(1).item())

    assert first == second
    assert first != third
