from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from torch.utils.tensorboard import SummaryWriter

from src.qrdqn_experiment import PreparedSplit, QRDQNRunConfig, run_qrdqn_experiment


class FakeQRDQN:
    def __init__(self, tensorboard_dir: Path, *, fail: bool = False) -> None:
        self.tensorboard_dir = Path(tensorboard_dir)
        self.fail = fail
        self.num_timesteps = 0

    def learn(self, *, total_timesteps: int, callback=None, tb_log_name: str, **_kwargs):
        if self.fail:
            raise RuntimeError("synthetic training failure")
        self.num_timesteps = total_timesteps + 8
        event_dir = self.tensorboard_dir / tb_log_name
        writer = SummaryWriter(log_dir=event_dir)
        writer.add_scalar("train/synthetic_reward", 1.0, self.num_timesteps)
        writer.add_scalar("train/learning_rate", 5e-5, self.num_timesteps)
        writer.add_scalar("train/loss", 0.25, self.num_timesteps)
        writer.flush()
        writer.close()
        if callback is not None:
            checkpoint_dir = Path(callback.save_path)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            for step in (1, 2, 3):
                (checkpoint_dir / f"synthetic_{step}_steps.zip").write_bytes(
                    f"checkpoint-{step}".encode()
                )
        return self

    def save(self, path: str) -> None:
        target = Path(path)
        if target.suffix != ".zip":
            target = target.with_suffix(".zip")
        target.write_bytes(b"synthetic-qrdqn-model")

    def predict(self, observations: np.ndarray, deterministic: bool = True):
        del deterministic
        return (observations[:, 0] > 0).astype(np.int64), None


@pytest.fixture
def synthetic_split() -> PreparedSplit:
    train_first = np.array([-6, -5, -4, 2, 3, 4], dtype=np.float32)
    test_first = np.array([-7, -1, 1, 8], dtype=np.float32)
    X_train = np.zeros((6, 152), dtype=np.float32)
    X_test = np.zeros((4, 152), dtype=np.float32)
    X_train[:, 0] = train_first
    X_test[:, 0] = test_first
    y_train = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    y_test = np.array([0, 0, 1, 1], dtype=np.int64)
    return PreparedSplit(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        feature_names=[f"feature_{index}" for index in range(152)],
        metadata={
            "split_mode": "random",
            "split_seed": 42,
            "cache_manifest_sha256": "a" * 64,
        },
    )


@pytest.fixture
def fake_model_factory():
    def factory(_config, _env, tensorboard_dir: Path, _device: str):
        return FakeQRDQN(tensorboard_dir)

    return factory


@pytest.fixture
def fresh_main_run(tmp_path: Path, synthetic_split: PreparedSplit, fake_model_factory) -> Path:
    config = QRDQNRunConfig(
        artifact_root=tmp_path / "runs",
        run_id="qrdqn_main_synthetic",
        dataset_root=tmp_path / "dataset",
        cache_root=tmp_path / "cache",
        cache_policy="require",
        split_mode="random",
        split_seed=42,
        model_seed=42,
        timesteps=2,
        checkpoint_freq=0,
        monitor_interval=0.01,
    )
    return run_qrdqn_experiment(
        config,
        split_loader=lambda _config: synthetic_split,
        model_factory=fake_model_factory,
    )
