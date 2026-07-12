#!/usr/bin/env python
"""Independent direct validation for one fresh schema-3 MAIN QRDQN run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
from sb3_contrib import QRDQN
from sklearn.metrics import confusion_matrix

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.artifact_integrity import (  # noqa: E402
    resolve_trusted_artifact,
    sha256_file,
    verify_artifact_manifest,
)
from src.cicids_cache import sha256_array  # noqa: E402
from src.metrics_utils import confusion_to_metrics  # noqa: E402
from src.qrdqn_experiment import (  # noqa: E402
    PreparedSplit,
    batched_predict,
    config_from_run_payload,
    load_experiment_split,
)
from src.run_artifacts import (  # noqa: E402
    ArtifactManifestWriter,
    ArtifactRequirement,
    TimingRecorder,
    atomic_write_json,
    atomic_write_text,
    collect_environment_metadata,
    tee_output,
)


SplitProvider = Callable[[dict[str, Any]], PreparedSplit]
ModelLoader = Callable[[Path], Any]


def _default_split_provider(source_config: dict[str, Any]) -> PreparedSplit:
    return load_experiment_split(config_from_run_payload(source_config))


def _default_model_loader(path: Path) -> Any:
    return QRDQN.load(str(path))


def _load_source(source_run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    verification = verify_artifact_manifest(source_run_dir)
    if verification["schema_version"] != "3.0":
        raise ValueError("Fresh MAIN validation requires a schema-3 source run")
    source_config = json.loads(
        (source_run_dir / "config.json").read_text(encoding="utf-8")
    )
    if source_config.get("profile_id") != "main-v1":
        raise ValueError("Fresh MAIN validation requires profile_id='main-v1'")
    return source_config, verification


def run_main_direct_validation(
    *,
    source_run_dir: Path,
    output_dir: Path,
    split_provider: SplitProvider = _default_split_provider,
    model_loader: ModelLoader = _default_model_loader,
    eval_batch_size: int = 8_192,
    campaign_id: str | None = None,
    logical_run_id: str | None = None,
    attempt: int = 1,
) -> Path:
    """Reproduce test labels, predict independently, and seal direct evidence."""
    source_run_dir = Path(source_run_dir)
    output_dir = Path(output_dir)
    if eval_batch_size <= 0:
        raise ValueError("eval_batch_size must be greater than zero")
    source_config, _verification = _load_source(source_run_dir)
    source_manifest_sha256 = sha256_file(source_run_dir / "artifact_manifest.json")
    run_metadata = {
        "campaign_id": campaign_id,
        "logical_run_id": logical_run_id or output_dir.name,
        "physical_run_id": output_dir.name,
        "attempt": attempt,
        "split_seed": source_config["split_seed"],
        "model_seed": source_config["model_seed"],
        "source_run_id": source_config["run_id"],
        "source_manifest_sha256": source_manifest_sha256,
    }
    writer = ArtifactManifestWriter(
        output_dir,
        run_metadata=run_metadata,
        requirements={
            "config": ArtifactRequirement("config.json"),
            "validation_results": ArtifactRequirement("validation_results.json"),
            "predictions": ArtifactRequirement("predictions.npz"),
            "environment": ArtifactRequirement("environment.json"),
            "timing": ArtifactRequirement("timing.json"),
            "stdout": ArtifactRequirement("stdout.log"),
            "stderr": ArtifactRequirement("stderr.log"),
        },
    )
    writer.start()
    atomic_write_text(output_dir / "stdout.log", "")
    atomic_write_text(output_dir / "stderr.log", "")
    timing = TimingRecorder()
    try:
        with tee_output(output_dir / "stdout.log", output_dir / "stderr.log"):
            model_path = resolve_trusted_artifact(
                source_run_dir, "model", repo_root=_REPO_ROOT
            )
            scaler_path = resolve_trusted_artifact(
                source_run_dir, "scaler", repo_root=_REPO_ROOT
            )
            split = split_provider(source_config)
            expected_hashes = source_config["split_metadata"]
            actual_hashes = {
                "test_set_sha256": sha256_array(split.X_test),
                "y_test_sha256": sha256_array(split.y_test),
            }
            for key, actual in actual_hashes.items():
                if actual != expected_hashes.get(key):
                    raise ValueError(
                        f"Reproduced {key} does not match fresh MAIN source metadata"
                    )

            scaler = joblib.load(scaler_path)
            with timing.measure("test_preprocessing") as measurement:
                X_test = scaler.transform(split.X_test).astype(np.float32)
                measurement.set_units(len(X_test), "rows")
            model = model_loader(model_path)
            with timing.measure("direct_prediction") as measurement:
                y_pred = batched_predict(model, X_test, eval_batch_size)
                measurement.set_units(len(y_pred), "rows")
            y_true = split.y_test.astype(np.int64)
            matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn, fp, fn, tp = (int(value) for value in matrix.ravel())
            reward_config = source_config["reward_config"]
            metrics = confusion_to_metrics(
                tn,
                fp,
                fn,
                tp,
                reward_config=reward_config,
                undefined_metric_policy="null",
            )
            results = {
                "source_run_id": source_config["run_id"],
                "source_manifest_sha256": source_manifest_sha256,
                "source_model_sha256": sha256_file(model_path),
                "source_scaler_sha256": sha256_file(scaler_path),
                "test_set_sha256": actual_hashes["test_set_sha256"],
                "y_test_sha256": actual_hashes["y_test_sha256"],
                "evaluation_basis": "direct_predictions_against_reproduced_test_labels",
                "environment_truth_metadata_used": False,
                "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
                "metrics": metrics,
            }
            np.savez_compressed(
                output_dir / "predictions.npz", y_true=y_true, y_pred=y_pred
            )
            atomic_write_json(output_dir / "validation_results.json", results)
            atomic_write_json(
                output_dir / "config.json",
                {
                    "job_type": "fresh_main_direct_validation",
                    "source_run_id": source_config["run_id"],
                    "source_manifest_sha256": source_manifest_sha256,
                    "split_seed": source_config["split_seed"],
                    "model_seed": source_config["model_seed"],
                    "eval_batch_size": eval_batch_size,
                },
            )
            atomic_write_json(
                output_dir / "environment.json",
                collect_environment_metadata(repo_root=_REPO_ROOT),
            )
            timing.write(output_dir / "timing.json")
        writer.complete()
        return output_dir
    except BaseException as error:
        try:
            timing.write(output_dir / "timing.json")
        except Exception:
            pass
        writer.fail(error)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independently validate a fresh schema-3 MAIN QRDQN run."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--eval-batch-size", type=int, default=8_192)
    parser.add_argument("--campaign-id", default=None)
    parser.add_argument("--logical-run-id", default=None)
    parser.add_argument("--attempt", type=int, default=1)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    run_main_direct_validation(
        source_run_dir=args.run_dir,
        output_dir=args.artifact_root / args.job_id,
        eval_batch_size=args.eval_batch_size,
        campaign_id=args.campaign_id,
        logical_run_id=args.logical_run_id,
        attempt=args.attempt,
    )


if __name__ == "__main__":
    main()
