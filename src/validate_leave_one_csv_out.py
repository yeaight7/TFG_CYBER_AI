"""Locked four-file targeted QRDQN holdout workflow for CICIDS2017.

The legacy filename remains for compatibility, but this module is not an
exhaustive leave-one-CSV-out runner. Omitting ``--holdout-csvs`` selects only
the four targeted campaign holdouts.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

if __package__ in {None, ""}:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from src.artifact_integrity import verify_artifact_manifest
    from src.load_cicids2017 import list_cicids2017_csv_files
    from src.metrics_utils import confusion_to_metrics
    from src.qrdqn_experiment import QRDQNRunConfig, run_qrdqn_experiment
    from src.run_artifacts import atomic_write_json
except ModuleNotFoundError:  # pragma: no cover - direct ``python src/...`` execution
    from artifact_integrity import verify_artifact_manifest
    from load_cicids2017 import list_cicids2017_csv_files
    from metrics_utils import confusion_to_metrics
    from qrdqn_experiment import QRDQNRunConfig, run_qrdqn_experiment
    from run_artifacts import atomic_write_json


TARGETED_QRDQN_HOLDOUTS = {
    "qrdqn_holdout_webattacks_m42":
        "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "qrdqn_holdout_infilteration_m42":
        "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "qrdqn_holdout_portscan_m42":
        "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "qrdqn_holdout_ddos_m42":
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
}

_TARGETED_FILENAMES = tuple(TARGETED_QRDQN_HOLDOUTS.values())
_MACRO_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "mcc",
    "precision_attack",
    "recall_attack",
    "f1_attack",
    "precision_benign",
    "recall_benign",
    "f1_benign",
    "specificity",
    "fpr",
    "fnr",
    "block_rate",
    "reward_total",
    "reward_per_sample",
)


@dataclass(frozen=True)
class TargetedHoldoutWorkflowConfig:
    artifact_root: Path
    dataset_root: Path
    cache_root: Path
    holdout_csvs: tuple[str, ...] | None = None
    profile_id: str = "main-v1"
    timesteps: int = 1_000_000
    split_seed: int = 42
    model_seed: int = 42
    cache_policy: str = "require"
    resume: bool = False
    checkpoint_freq: int | None = None
    checkpoint_keep: int = 2
    monitor_interval: float = 30.0
    eval_batch_size: int = 8_192

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_root", Path(self.artifact_root))
        object.__setattr__(self, "dataset_root", Path(self.dataset_root))
        object.__setattr__(self, "cache_root", Path(self.cache_root))
        if self.profile_id != "main-v1":
            raise ValueError("Targeted holdouts require profile 'main-v1'")
        if self.model_seed != 42:
            raise ValueError("The locked targeted holdout study requires model_seed=42")
        if self.timesteps <= 0:
            raise ValueError("timesteps must be greater than zero")
        selected = self.holdout_csvs or _TARGETED_FILENAMES
        if len(selected) != len(set(selected)):
            raise ValueError("duplicate targeted holdout filename")
        unknown = [name for name in selected if name not in _TARGETED_FILENAMES]
        if unknown:
            raise ValueError(f"Unknown targeted holdout filename: {unknown}")

    @property
    def summary_path(self) -> Path:
        return self.artifact_root / "qrdqn_targeted_holdouts_summary.json"


HoldoutExecutor = Callable[[QRDQNRunConfig], Path]


def resolve_targeted_holdouts(
    requested_csvs: Sequence[str] | None,
    available_csvs: Sequence[str],
) -> list[str]:
    """Resolve an exact subset of the locked four holdouts, never all eight."""
    requested = list(_TARGETED_FILENAMES if requested_csvs is None else requested_csvs)
    if len(requested) != len(set(requested)):
        raise ValueError("duplicate targeted holdout filename")

    available = set(available_csvs)
    resolved: list[str] = []
    for name in requested:
        if name not in _TARGETED_FILENAMES:
            if name.lower() in {target.lower() for target in _TARGETED_FILENAMES}:
                raise ValueError(f"Targeted holdout filenames must match exactly: {name}")
            raise ValueError(f"Unknown targeted holdout filename: {name}")
        if name not in available:
            raise ValueError(f"Targeted holdout is not available in the dataset: {name}")
        resolved.append(name)
    return resolved


def aggregate_holdout_metrics(fold_metrics: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Compute defined-only macro summaries and separately labelled pooled metrics."""
    if not fold_metrics:
        raise ValueError("At least one completed holdout is required for aggregation")

    macro: dict[str, dict[str, float | int | None]] = {}
    for metric_name in _MACRO_METRICS:
        defined = [
            float(metrics[metric_name])
            for metrics in fold_metrics
            if metrics.get(metric_name) is not None
        ]
        if defined or any(metric_name in metrics for metrics in fold_metrics):
            macro[metric_name] = {
                "mean": None if not defined else float(sum(defined) / len(defined)),
                "n_defined": len(defined),
            }

    pooled = {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
    for metrics in fold_metrics:
        matrix = metrics.get("confusion_matrix")
        if not isinstance(matrix, dict) or set(matrix) != set(pooled):
            raise ValueError("Each holdout must contain a labelled confusion_matrix")
        for count_name in pooled:
            pooled[count_name] += int(matrix[count_name])

    pooled_metrics = confusion_to_metrics(
        pooled["tn"],
        pooled["fp"],
        pooled["fn"],
        pooled["tp"],
        undefined_metric_policy="null",
    )
    return {
        "n_holdouts": len(fold_metrics),
        "defined_only_macro": macro,
        "pooled_confusion_matrix": pooled,
        "pooled_metrics": pooled_metrics,
    }


def _read_completed_metrics(run_dir: Path) -> dict[str, Any]:
    verify_artifact_manifest(run_dir)
    return json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))


def run_targeted_holdouts(
    config: TargetedHoldoutWorkflowConfig,
    *,
    executor: HoldoutExecutor = run_qrdqn_experiment,
) -> dict[str, Any]:
    """Run or resume the selected locked holdouts, sealing each run immediately."""
    selected = list(config.holdout_csvs or _TARGETED_FILENAMES)
    filename_to_run_id = {filename: run_id for run_id, filename in TARGETED_QRDQN_HOLDOUTS.items()}
    records: list[dict[str, Any]] = []
    completed_metrics: list[dict[str, Any]] = []

    for holdout_csv in selected:
        run_id = filename_to_run_id[holdout_csv]
        run_dir = config.artifact_root / run_id
        execution = "completed"
        if run_dir.exists() and config.resume:
            metrics = _read_completed_metrics(run_dir)
            execution = "skipped_completed"
        else:
            run_config = QRDQNRunConfig(
                artifact_root=config.artifact_root,
                run_id=run_id,
                dataset_root=config.dataset_root,
                cache_root=config.cache_root,
                cache_policy=config.cache_policy,
                split_mode="exact-holdout",
                split_seed=config.split_seed,
                model_seed=config.model_seed,
                profile_id=config.profile_id,
                timesteps=config.timesteps,
                holdout_csv=holdout_csv,
                checkpoint_freq=config.checkpoint_freq,
                checkpoint_keep=config.checkpoint_keep,
                monitor_interval=config.monitor_interval,
                eval_batch_size=config.eval_batch_size,
                logical_run_id=run_id,
            )
            completed_dir = executor(run_config)
            if Path(completed_dir).resolve() != run_dir.resolve():
                raise ValueError("Targeted holdout executor returned an unexpected run directory")
            metrics = _read_completed_metrics(run_dir)

        completed_metrics.append(metrics)
        records.append(
            {
                "logical_run_id": run_id,
                "holdout_csv": holdout_csv,
                "execution": execution,
                "run_dir": str(run_dir),
            }
        )
        summary = {
            "workflow": "targeted_four_holdout_qrdqn",
            "profile_id": config.profile_id,
            "timesteps": config.timesteps,
            "split_seed": config.split_seed,
            "model_seed": config.model_seed,
            "holdout_csvs": selected,
            "runs": records,
            "aggregate": aggregate_holdout_metrics(completed_metrics),
        }
        atomic_write_json(config.summary_path, summary)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Locked four-file targeted QRDQN holdout workflow for CICIDS2017",
    )
    parser.add_argument("--holdout-csvs", nargs="+", default=None)
    parser.add_argument("--profile", default="main-v1", choices=["main-v1"])
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/CICIDS2017"))
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--cache-policy", choices=["off", "prefer", "require"], default="require")
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-freq", type=int, default=None)
    parser.add_argument("--checkpoint-keep", type=int, default=2)
    parser.add_argument("--monitor-interval", type=float, default=30.0)
    parser.add_argument("--eval-batch-size", type=int, default=8_192)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    available = [path.name for path in list_cicids2017_csv_files(args.dataset_root)]
    selected = resolve_targeted_holdouts(args.holdout_csvs, available)
    config = TargetedHoldoutWorkflowConfig(
        artifact_root=args.artifact_root,
        dataset_root=args.dataset_root,
        cache_root=args.cache_root,
        cache_policy=args.cache_policy,
        holdout_csvs=tuple(selected),
        profile_id=args.profile,
        timesteps=args.timesteps,
        split_seed=args.split_seed,
        model_seed=args.model_seed,
        resume=args.resume,
        checkpoint_freq=args.checkpoint_freq,
        checkpoint_keep=args.checkpoint_keep,
        monitor_interval=args.monitor_interval,
        eval_batch_size=args.eval_batch_size,
    )
    summary = run_targeted_holdouts(config)
    print(f"Targeted holdouts complete: {len(summary['runs'])}")
    print(f"Summary: {config.summary_path}")


if __name__ == "__main__":
    main()
