import argparse
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from train_rl_defender import (  # noqa: E402
    resolve_checkpoint_freq,
    resolve_total_timesteps,
    resolve_training_hyperparams,
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
