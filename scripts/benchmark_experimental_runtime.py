"""Isolated synthetic runtime benchmark for GPU-host thread configurations."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

if __package__ in {None, ""}:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.run_artifacts import atomic_write_json  # noqa: E402


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def parse_thread_config(value: str) -> tuple[int, int]:
    parts = value.split(":")
    if len(parts) != 2:
        raise ValueError("Thread config must use TORCH_THREADS:INTEROP_THREADS")
    try:
        torch_threads, interop_threads = (int(part) for part in parts)
    except ValueError as error:
        raise ValueError("Thread config values must be integers") from error
    if torch_threads <= 0 or interop_threads <= 0:
        raise ValueError("Thread config values must be greater than zero")
    return torch_threads, interop_threads


def _synchronise(torch: Any, device: Any) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_worker(thread_config: str) -> dict[str, Any]:
    torch_threads, interop_threads = parse_thread_config(thread_config)
    os.environ["OMP_NUM_THREADS"] = str(torch_threads)
    os.environ["MKL_NUM_THREADS"] = str(torch_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(torch_threads)

    import numpy as np
    import torch
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    torch.set_num_threads(torch_threads)
    torch.set_num_interop_threads(interop_threads)
    rng = np.random.default_rng(42)
    rows = 4_096
    features = 152
    X = rng.normal(size=(rows, features)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.int64)

    started = time.perf_counter()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)
    preprocessing_seconds = time.perf_counter() - started

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = torch.nn.Sequential(
        torch.nn.Linear(features, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 2),
    ).to(device)
    optimiser = torch.optim.Adam(network.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    train_X = torch.from_numpy(X_scaled[:2_048]).to(device)
    train_y = torch.from_numpy(y[:2_048]).to(device)
    iterations = 8
    _synchronise(torch, device)
    started = time.perf_counter()
    for _ in range(iterations):
        optimiser.zero_grad(set_to_none=True)
        loss = loss_fn(network(train_X), train_y)
        loss.backward()
        optimiser.step()
    _synchronise(torch, device)
    training_seconds = time.perf_counter() - started

    eval_X = torch.from_numpy(X_scaled).to(device)
    _synchronise(torch, device)
    started = time.perf_counter()
    with torch.inference_mode():
        network(eval_X)
    _synchronise(torch, device)
    evaluation_seconds = time.perf_counter() - started

    forest = RandomForestClassifier(
        n_estimators=10,
        max_depth=8,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    started = time.perf_counter()
    forest.fit(X_scaled, y)
    rf_fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    forest.predict(X_scaled)
    rf_predict_seconds = time.perf_counter() - started

    return {
        "status": "completed",
        "workload": "synthetic-runtime-probe-v1",
        "scientific_result": False,
        "requested": {
            "torch_threads": torch_threads,
            "torch_inter_op_threads": interop_threads,
            "OMP_NUM_THREADS": str(torch_threads),
            "MKL_NUM_THREADS": str(torch_threads),
            "OPENBLAS_NUM_THREADS": str(torch_threads),
        },
        "effective": {
            "torch_threads": int(torch.get_num_threads()),
            "torch_inter_op_threads": int(torch.get_num_interop_threads()),
            "device": str(device),
        },
        "measurements": {
            "preprocessing_seconds": preprocessing_seconds,
            "preprocessing_rows_per_second": rows / preprocessing_seconds,
            "synthetic_qrdqn_training_seconds": training_seconds,
            "synthetic_qrdqn_update_steps_per_second": iterations / training_seconds,
            "synthetic_qrdqn_training_rows_per_second": (iterations * len(train_X))
            / training_seconds,
            "evaluation_seconds": evaluation_seconds,
            "evaluation_rows_per_second": rows / evaluation_seconds,
            "random_forest_fit_seconds": rf_fit_seconds,
            "random_forest_prediction_seconds": rf_predict_seconds,
            "random_forest_prediction_rows_per_second": rows / rf_predict_seconds,
        },
    }


def run_runtime_benchmark(
    *,
    output_path: Path | str,
    thread_configs: Sequence[str],
    command_runner: CommandRunner = subprocess.run,
) -> dict[str, Any]:
    parsed = [parse_thread_config(value) for value in thread_configs]
    if len(set(parsed)) != len(parsed):
        raise ValueError("Thread configs must be unique")
    results: list[dict[str, Any]] = []
    script_path = Path(__file__).resolve()
    for original, (torch_threads, interop_threads) in zip(thread_configs, parsed, strict=True):
        normalised = f"{torch_threads}:{interop_threads}"
        command = [sys.executable, str(script_path), "--worker-config", normalised]
        completed = command_runner(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Runtime benchmark worker failed for {original}: {completed.stderr.strip()}"
            )
        try:
            result = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"Runtime benchmark worker returned invalid JSON for {original}"
            ) from error
        if not isinstance(result, dict) or result.get("status") != "completed":
            raise RuntimeError(f"Runtime benchmark worker did not complete for {original}")
        results.append(result)

    report = {
        "schema_version": "1.0",
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "execution_model": "one isolated subprocess per thread configuration",
        "scientific_config_selected": False,
        "results": results,
    }
    atomic_write_json(Path(output_path), report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark provider-neutral GPU-host thread configurations with small "
            "synthetic workloads; no scientific configuration is selected."
        )
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--thread-config", nargs="+", default=["1:1", "4:1", "8:1", "16:1"])
    parser.add_argument("--worker-config", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker_config is None and args.output is None:
        parser.error("--output is required")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker_config is not None:
        try:
            result = run_worker(args.worker_config)
        except (RuntimeError, ValueError) as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
        print(json.dumps(result, sort_keys=True))
        return 0
    try:
        report = run_runtime_benchmark(
            output_path=args.output,
            thread_configs=args.thread_config,
        )
    except (RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
