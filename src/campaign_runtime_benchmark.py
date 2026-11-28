"""Optional timing of the real QRDQN campaign path on bounded synthetic training data."""

from __future__ import annotations

import contextlib
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import torch
from sb3_contrib import QRDQN
from sklearn.preprocessing import StandardScaler
from stable_baselines3.common.logger import configure

from src.artifact_integrity import resolve_trusted_artifact, sha256_file
from src.cicids_cache import sha256_array
from src.experiment_profiles import MAIN_V1_PROFILE
from src.qrdqn_experiment import (
    QRDQNRunConfig,
    _make_environment,
    batched_predict,
    create_qrdqn_model,
    seed_model_rngs,
)
from src.run_artifacts import TimingRecorder, collect_environment_metadata, configure_torch_runtime


SEED = 42
REPEATS = 3
REPO_ROOT = Path(__file__).resolve().parents[1]


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class _TimedQRDQN(QRDQN):
    """Use the stock learn loop and separate time spent collecting and updating."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.windows: dict[str, list[tuple[int, float]]] = defaultdict(list)

    def collect_rollouts(self, *args, **kwargs):
        phase = "warmup_rollout" if self.num_timesteps < self.learning_starts else "learned_rollout"
        started = perf_counter()
        rollout = super().collect_rollouts(*args, **kwargs)
        _synchronize(self.device)
        self.windows[phase].append((rollout.episode_timesteps, perf_counter() - started))
        return rollout

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        started = perf_counter()
        super().train(gradient_steps, batch_size)
        _synchronize(self.device)
        self.windows["optimizer_updates"].append((gradient_steps, perf_counter() - started))


def _rates(records: list[tuple[int, float]]) -> dict:
    rates = [units / elapsed for units, elapsed in records]
    units = sum(units for units, _ in records)
    seconds = sum(elapsed for _, elapsed in records)
    return {
        "units": units,
        "duration_seconds": seconds,
        "throughput_per_second": units / seconds,
        "median_window_throughput": median(rates),
        "window_throughput_range": [min(rates), max(rates)],
        "windows": len(records),
    }


def _inference_sweep(model: QRDQN, X: np.ndarray, batch_sizes: list[int]) -> dict:
    references = {batch: batched_predict(model, X, batch) for batch in (4_096, 8_192)}
    results = {}
    for batch in dict.fromkeys((*batch_sizes, *references)):
        try:
            batched_predict(model, X, batch)
            timings = []
            mismatches = dict.fromkeys(references, 0)
            for _ in range(REPEATS):
                _synchronize(model.device)
                started = perf_counter()
                actions = batched_predict(model, X, batch)
                _synchronize(model.device)
                timings.append(perf_counter() - started)
                for reference_batch, expected in references.items():
                    mismatches[reference_batch] = max(
                        mismatches[reference_batch], int(np.count_nonzero(actions != expected))
                    )
            results[batch] = {
                "status": "completed",
                "median_rows_per_second": len(X) / median(timings),
                "seconds": timings,
                "mismatches_by_reference_batch": mismatches,
            }
        except torch.cuda.OutOfMemoryError:
            results[batch] = {"status": "cuda_out_of_memory"}
            torch.cuda.empty_cache()
    return {
        "rows": len(X),
        "observation_sha256": sha256_array(X),
        "reference_prediction_sha256": {
            batch: sha256_array(actions) for batch, actions in references.items()
        },
        "batches": results,
    }


def run_campaign_worker(
    *,
    torch_threads: int,
    interop_threads: int,
    device_request: str,
    batch_sizes: list[int],
    inference_rows: int,
    inference_run_dir: Path | None,
    observations_path: Path | None,
    run_dir: Path,
) -> dict:
    runtime = configure_torch_runtime(torch_threads, interop_threads)
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if device_request == "auto"
        else device_request
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested and is unavailable")
    if runtime["float32_matmul_precision"] != "highest":
        raise ValueError("Compare campaign runtimes with full FP32 matmul precision")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    timing = TimingRecorder()
    profile = MAIN_V1_PROFILE.qrdqn_hyperparams()
    timesteps = profile["learning_starts"] + (REPEATS + 1) * profile["train_freq"]
    run_config = QRDQNRunConfig(
        artifact_root=run_dir,
        run_id="benchmark",
        dataset_root=run_dir,
        cache_root=None,
        cache_policy="off",
        timesteps=timesteps,
        torch_threads=torch_threads,
        torch_inter_op_threads=interop_threads,
        verbose=0,
    )
    seed_model_rngs(SEED)
    rng = np.random.default_rng(SEED)
    with timing.measure("synthetic_preprocessing") as phase:
        X_train = rng.standard_normal((20_000, 152), dtype=np.float32)
        y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(np.int64)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        phase.set_units(len(X_train), "rows")
    env = _make_environment(X_train, y_train, MAIN_V1_PROFILE.reward_config())
    try:
        env.seed(SEED)
        model = create_qrdqn_model(run_config, env, run_dir, str(device), model_class=_TimedQRDQN)
        model.set_logger(configure(format_strings=[]))
        with timing.measure("training_wall_clock") as phase:
            _synchronize(device)
            model.learn(total_timesteps=timesteps, reset_num_timesteps=False)
            _synchronize(device)
            phase.set_units(model.num_timesteps, "timesteps")
        phases = {
            name: _rates(windows[1:] if name != "warmup_rollout" else windows)
            for name, windows in model.windows.items()
        }
        update_count = model._n_updates
    finally:
        env.close()

    with timing.measure("inference_input_preparation") as phase:
        if observations_path is None:
            X_eval = scaler.transform(rng.standard_normal((inference_rows, 152), dtype=np.float32))
        else:
            X_eval = np.load(observations_path, allow_pickle=False)
        if (
            X_eval.dtype != np.float32
            or X_eval.ndim != 2
            or X_eval.shape[1] != 152
            or not len(X_eval)
        ):
            raise ValueError("Expected a non-empty float32 .npy array with 152 observation columns")
        phase.set_units(len(X_eval), "rows")
    checkpoint_sha256 = None
    if inference_run_dir is not None:
        checkpoint = resolve_trusted_artifact(inference_run_dir, "model", repo_root=REPO_ROOT)
        checkpoint_sha256 = sha256_file(checkpoint)
        model = QRDQN.load(str(checkpoint), device=device)
    inference = _inference_sweep(model, X_eval, batch_sizes)
    host_peak_rss = None
    with contextlib.suppress(ImportError):
        import resource

        host_peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform.startswith("linux"):
            host_peak_rss *= 1_024
    return {
        "status": "completed",
        "workload": "campaign-qrdqn-runtime-v1",
        "scientific_result": False,
        "profile_hash": MAIN_V1_PROFILE.content_hash,
        "seed": SEED,
        "device": str(device),
        "runtime": runtime,
        "environment": collect_environment_metadata(
            repo_root=REPO_ROOT,
            requested_torch_threads=torch_threads,
            requested_torch_interop_threads=interop_threads,
        ),
        "peak_process_rss_bytes": host_peak_rss,
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "timing": timing.to_dict(),
        "phases": phases,
        "total_optimizer_updates": update_count,
        "inference_checkpoint_sha256": checkpoint_sha256,
        "inference": inference,
    }
