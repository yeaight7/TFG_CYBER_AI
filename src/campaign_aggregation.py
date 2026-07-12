"""Validated campaign aggregation and future figure generation.

The module consumes only a completed schema-1 campaign whose schema-3 run
manifests still verify.  It never reads historical result directories and it
keeps auxiliary controls and Phase 2 laboratory inference outside primary
CICIDS2017 performance groups.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from html import escape
import io
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence
import uuid

from src.artifact_integrity import (
    ArtifactTrustError,
    artifact_manifest_entry,
    load_artifact_manifest,
    sha256_file,
    verify_artifact_manifest,
)
from src.campaign import (
    CAMPAIGN_STATE_SCHEMA_VERSION,
    FRESH_MAIN_ID,
    CampaignEntry,
    CampaignSpec,
    CampaignSpecError,
    validate_campaign_spec,
)
from src.experiment_profiles import MAIN_V1_PROFILE_HASH
from src.metrics_utils import confusion_to_metrics
from src.run_artifacts import atomic_write_json, atomic_write_text


AGGREGATE_SCHEMA_VERSION = "1.0"
FIGURE_SCHEMA_VERSION = "1.0"
_REPO_ROOT = Path(__file__).resolve().parent.parent

AGGREGATE_DATA_FILES = (
    "main.json",
    "main_direct_validation.json",
    "main_bootstrap_ci.json",
    "main_duplicate_analysis.json",
    "shuffled_label_validation.json",
    "phase2_fresh_main.json",
    "day_split.json",
    "size_ladder.json",
    "size_ladder.csv",
    "seed_sensitivity.json",
    "seed_sensitivity.csv",
    "targeted_holdouts.json",
    "targeted_holdouts.csv",
    "random_forest.json",
    "random_forest.csv",
    "qrdqn_vs_rf.csv",
)
AGGREGATE_FILES = ("campaign_summary.json", *AGGREGATE_DATA_FILES)
FIGURE_FILES = (
    "size_ladder.svg",
    "seed_sensitivity.svg",
    "day_generalisation.svg",
    "targeted_holdouts.svg",
    "qrdqn_vs_rf.svg",
)

_REQUIRED_CAMPAIGN_FILES = (
    "campaign_state.json",
    "campaign_spec_original.json",
    "campaign_spec_resolved.json",
    "preflight_report.json",
    "cache_manifest.json",
)
_HASH_FIELDS = (
    "train_set_sha256",
    "y_train_sha256",
    "test_set_sha256",
    "y_test_sha256",
)
_SUMMARY_METRICS = (
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
)
_CSV_FIELDS = (
    "campaign_id",
    "logical_run_id",
    "physical_run_id",
    "artifact_dir",
    "manifest_sha256",
    "classification",
    "stage",
    "model_family",
    "reuse_of",
    "source_logical_run_id",
    "split_mode",
    "split_seed",
    "model_seed",
    "timesteps",
    "train_max_rows",
    "train_rows",
    "holdout_csv",
    "campaign_profile_id",
    "campaign_profile_hash",
    "profile_id",
    "profile_hash",
    "train_set_sha256",
    "y_train_sha256",
    "test_set_sha256",
    "y_test_sha256",
    "support_n_test",
    "support_benign",
    "support_attack",
    "attack_prevalence",
    *_SUMMARY_METRICS,
    "tp",
    "tn",
    "fp",
    "fn",
    "timings_json",
)


class CampaignAggregationError(RuntimeError):
    """Campaign evidence or aggregate output is unsafe or incompatible."""


@dataclass(frozen=True)
class _RunEvidence:
    entry: CampaignEntry
    record: Mapping[str, Any]
    artifact_dir: Path
    artifact_relative: str
    manifest: Mapping[str, Any]
    manifest_sha256: str
    config: Mapping[str, Any]
    metrics: Mapping[str, Any] | None
    timing: Mapping[str, Any] | None
    reuse_of: str | None = None
    source: _RunEvidence | None = None

    @property
    def physical(self) -> _RunEvidence:
        return self.source or self


@dataclass(frozen=True)
class _CampaignEvidence:
    campaign_dir: Path
    campaign_id: str
    spec: CampaignSpec
    state: Mapping[str, Any]
    resolved_spec: Mapping[str, Any]
    runs: Mapping[str, _RunEvidence]
    auxiliary_payloads: Mapping[str, Mapping[str, Any]]


def _read_json(path: Path, *, label: str | None = None) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CampaignAggregationError(
            f"Cannot read {label or 'JSON evidence'}: {path}"
        ) from error
    if not isinstance(value, dict):
        raise CampaignAggregationError(f"Expected JSON object for {label or path.name}")
    return value


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _safe_relative(root: Path, value: Any, *, field: str) -> tuple[Path, str]:
    if not isinstance(value, str):
        raise CampaignAggregationError(f"Invalid {field}: {value!r}")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise CampaignAggregationError(f"Unsafe {field}: {value!r}")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise CampaignAggregationError(f"Unsafe {field}: {value!r}") from error
    return resolved, relative.as_posix()


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _validate_output_path(
    output_dir: Path,
    *,
    repo_root: Path,
    source_dir: Path | None = None,
) -> Path:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise CampaignAggregationError(f"Output directory already exists: {output_dir}")
    for protected_name in ("memoria", "report"):
        protected = (repo_root / protected_name).resolve()
        if _inside(output_dir, protected):
            raise CampaignAggregationError(
                f"Output path is inside protected {protected_name}/: {output_dir}"
            )
    if source_dir is not None and _inside(output_dir, source_dir):
        raise CampaignAggregationError("Output directory must be outside source evidence")
    return output_dir


def _write_directory_atomically(output_dir: Path, writer) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dir.with_name(f".{output_dir.name}.{uuid.uuid4().hex}.tmp")
    temporary.mkdir()
    try:
        writer(temporary)
        os.replace(temporary, output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _trusted_path(evidence: _RunEvidence, artifact_name: str) -> Path:
    try:
        entry = artifact_manifest_entry(evidence.manifest, artifact_name)
    except ArtifactTrustError as error:
        raise CampaignAggregationError(
            f"{evidence.entry.logical_id} lacks required artifact {artifact_name}"
        ) from error
    relative = entry.get("relative_path")
    if not isinstance(relative, str):
        raise CampaignAggregationError(
            f"{evidence.entry.logical_id} has invalid artifact {artifact_name}"
        )
    path, _relative = _safe_relative(
        evidence.artifact_dir,
        relative,
        field=f"{artifact_name} path for {evidence.entry.logical_id}",
    )
    if not path.is_file():
        raise CampaignAggregationError(f"Required artifact disappeared: {path}")
    return path


def _trusted_json(evidence: _RunEvidence, artifact_name: str) -> dict[str, Any]:
    return _read_json(
        _trusted_path(evidence, artifact_name),
        label=f"{artifact_name} for {evidence.entry.logical_id}",
    )


def _maybe_trusted_json(
    evidence: _RunEvidence,
    artifact_name: str,
) -> dict[str, Any] | None:
    file_artifacts = evidence.manifest.get("file_artifacts")
    if not isinstance(file_artifacts, Mapping) or artifact_name not in file_artifacts:
        return None
    return _trusted_json(evidence, artifact_name)


def _resolved_config_value(config: Mapping[str, Any], key: str) -> Any:
    if key in config:
        return config[key]
    request = config.get("request")
    return request.get(key) if isinstance(request, Mapping) else None


def _validate_metric_values(
    metrics: Mapping[str, Any],
    confusion: Mapping[str, Any],
    *,
    logical_id: str,
) -> None:
    counts: dict[str, int] = {}
    for key in ("tn", "fp", "fn", "tp"):
        value = confusion.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise CampaignAggregationError(f"Invalid confusion matrix for {logical_id}")
        counts[key] = value
    expected = confusion_to_metrics(
        counts["tn"],
        counts["fp"],
        counts["fn"],
        counts["tp"],
        undefined_metric_policy="null",
    )
    for key in (*_SUMMARY_METRICS, "tn", "fp", "fn", "tp"):
        actual = metrics.get(key)
        wanted = expected[key]
        if wanted is None:
            if actual is not None:
                raise CampaignAggregationError(
                    f"Undefined metric policy mismatch for {logical_id}: {key}"
                )
        elif not isinstance(actual, (int, float)) or isinstance(actual, bool):
            raise CampaignAggregationError(f"Missing metric for {logical_id}: {key}")
        elif not math.isclose(float(actual), float(wanted), rel_tol=1e-9, abs_tol=1e-12):
            raise CampaignAggregationError(f"Metric mismatch for {logical_id}: {key}")


def _validate_primary_payload(evidence: _RunEvidence) -> None:
    entry = evidence.entry
    config = evidence.config
    expected = {
        "split_mode": entry.config.get("split_mode"),
        "split_seed": entry.config.get("split_seed"),
        "model_seed": entry.config.get("model_seed"),
        "train_max_rows": entry.config.get("train_max_rows"),
        "holdout_csv": entry.config.get("holdout_csv"),
    }
    if entry.runner == "qrdqn":
        expected.update(
            {
                "timesteps": entry.config.get("timesteps"),
                "profile_id": "main-v1",
            }
        )
    mismatched = {
        key: value
        for key, value in expected.items()
        if _resolved_config_value(config, key) != value
    }
    if mismatched:
        raise CampaignAggregationError(
            f"Incompatible scientific config for {entry.logical_id}: {mismatched}"
        )
    run = evidence.manifest.get("run")
    if not isinstance(run, Mapping):
        raise CampaignAggregationError(f"Invalid manifest run identity: {entry.logical_id}")
    if entry.runner == "qrdqn":
        if (
            config.get("profile_id") != "main-v1"
            or config.get("profile_hash") != MAIN_V1_PROFILE_HASH
            or run.get("profile_id") != "main-v1"
            or run.get("profile_hash") != MAIN_V1_PROFILE_HASH
        ):
            raise CampaignAggregationError(f"Profile mismatch for {entry.logical_id}")
    elif config.get("algorithm") != "RandomForest" or run.get("algorithm") != "RandomForest":
        raise CampaignAggregationError(f"Algorithm mismatch for {entry.logical_id}")

    split = config.get("split_metadata")
    if not isinstance(split, Mapping):
        raise CampaignAggregationError(f"Missing split metadata for {entry.logical_id}")
    for key in _HASH_FIELDS:
        if not _is_sha256(split.get(key)):
            raise CampaignAggregationError(f"Invalid {key} for {entry.logical_id}")
    if not isinstance(evidence.metrics, Mapping):
        raise CampaignAggregationError(f"Missing metrics for {entry.logical_id}")
    confusion = evidence.metrics.get("confusion_matrix")
    support = evidence.metrics.get("support")
    if not isinstance(confusion, Mapping) or not isinstance(support, Mapping):
        raise CampaignAggregationError(f"Incomplete metrics for {entry.logical_id}")
    _validate_metric_values(evidence.metrics, confusion, logical_id=entry.logical_id)
    n_test = sum(int(confusion[key]) for key in ("tn", "fp", "fn", "tp"))
    if (
        support.get("n_test") != n_test
        or support.get("benign") != int(confusion["tn"]) + int(confusion["fp"])
        or support.get("attack") != int(confusion["fn"]) + int(confusion["tp"])
    ):
        raise CampaignAggregationError(f"Support mismatch for {entry.logical_id}")
    if not isinstance(evidence.timing, Mapping) or not isinstance(
        evidence.timing.get("phases"), Mapping
    ):
        raise CampaignAggregationError(f"Missing timings for {entry.logical_id}")


def _load_physical_evidence(
    campaign_dir: Path,
    campaign_id: str,
    entry: CampaignEntry,
    record: Mapping[str, Any],
) -> _RunEvidence:
    attempts = record.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise CampaignAggregationError(f"Completed entry has no attempt: {entry.logical_id}")
    attempt_record = attempts[-1]
    if not isinstance(attempt_record, Mapping) or attempt_record.get("status") != "completed":
        raise CampaignAggregationError(f"Completed entry has invalid attempt: {entry.logical_id}")
    attempt_number = attempt_record.get("attempt")
    if not isinstance(attempt_number, int) or isinstance(attempt_number, bool):
        raise CampaignAggregationError(f"Invalid attempt number: {entry.logical_id}")
    artifact_dir, artifact_relative = _safe_relative(
        campaign_dir,
        attempt_record.get("artifact_dir"),
        field=f"artifact_dir for {entry.logical_id}",
    )
    try:
        verification = verify_artifact_manifest(artifact_dir)
        manifest = load_artifact_manifest(artifact_dir)
    except (ArtifactTrustError, OSError, ValueError, json.JSONDecodeError) as error:
        raise CampaignAggregationError(
            f"Invalid campaign artifact for {entry.logical_id}: {error}"
        ) from error
    if verification.get("schema_version") != "3.0":
        raise CampaignAggregationError(
            f"Fresh campaign requires schema-3 evidence: {entry.logical_id}"
        )
    manifest_sha256 = sha256_file(artifact_dir / "artifact_manifest.json")
    if attempt_record.get("manifest_sha256") != manifest_sha256:
        raise CampaignAggregationError(f"Manifest state mismatch for {entry.logical_id}")
    run = manifest.get("run")
    expected_identity = {
        "campaign_id": campaign_id,
        "logical_run_id": entry.logical_id,
        "physical_run_id": attempt_record.get("physical_run_id"),
        "attempt": attempt_number,
    }
    if not isinstance(run, Mapping) or any(
        run.get(key) != value for key, value in expected_identity.items()
    ):
        raise CampaignAggregationError(f"Manifest identity mismatch for {entry.logical_id}")

    provisional = _RunEvidence(
        entry=entry,
        record=record,
        artifact_dir=artifact_dir,
        artifact_relative=artifact_relative,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        config={},
        metrics=None,
        timing=None,
    )
    config = _trusted_json(provisional, "config")
    evidence = _RunEvidence(
        entry=entry,
        record=record,
        artifact_dir=artifact_dir,
        artifact_relative=artifact_relative,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        config=config,
        metrics=_maybe_trusted_json(provisional, "metrics"),
        timing=_maybe_trusted_json(provisional, "timing"),
    )
    if entry.classification == "primary_model_training":
        _validate_primary_payload(evidence)
    return evidence


def _require_fresh_binding(evidence: _RunEvidence, main: _RunEvidence) -> None:
    main_run = main.manifest.get("run")
    run = evidence.manifest.get("run")
    if not isinstance(main_run, Mapping) or not isinstance(run, Mapping):
        raise CampaignAggregationError("Fresh campaign MAIN has invalid identity")
    expected = {
        "source_run_id": main_run.get("physical_run_id"),
        "source_manifest_sha256": main.manifest_sha256,
    }
    if any(run.get(key) != value for key, value in expected.items()) or any(
        evidence.config.get(key) != value for key, value in expected.items()
    ):
        raise CampaignAggregationError(
            f"{evidence.entry.logical_id} is not bound to the fresh campaign MAIN"
        )


def _manifest_artifact_sha(evidence: _RunEvidence, name: str) -> str:
    try:
        value = artifact_manifest_entry(evidence.manifest, name).get("sha256")
    except ArtifactTrustError as error:
        raise CampaignAggregationError(f"Fresh MAIN lacks {name} provenance") from error
    if not _is_sha256(value):
        raise CampaignAggregationError(f"Fresh MAIN has invalid {name} provenance")
    return str(value)


def _validate_auxiliary_payloads(
    runs: Mapping[str, _RunEvidence],
    resolved_spec: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    main = runs[FRESH_MAIN_ID]
    main_run = main.manifest["run"]
    source_run_id = main_run["physical_run_id"]
    main_split = main.config.get("split_metadata")
    if not isinstance(main_split, Mapping):
        raise CampaignAggregationError("Fresh campaign MAIN lacks split metadata")
    source_hashes = {
        "model": _manifest_artifact_sha(main, "model"),
        "scaler": _manifest_artifact_sha(main, "scaler"),
        "predictions": _manifest_artifact_sha(main, "predictions"),
        "train_percentiles": _manifest_artifact_sha(main, "train_percentiles"),
        "feature_names": _manifest_artifact_sha(main, "feature_names"),
        "manifest": main.manifest_sha256,
    }
    payloads: dict[str, Mapping[str, Any]] = {}

    direct = runs["main_direct_validation"]
    _require_fresh_binding(direct, main)
    direct_result = _trusted_json(direct, "validation_results")
    direct_expected = {
        "source_run_id": source_run_id,
        "source_manifest_sha256": main.manifest_sha256,
        "source_model_sha256": source_hashes["model"],
        "source_scaler_sha256": source_hashes["scaler"],
        "test_set_sha256": main_split.get("test_set_sha256"),
        "y_test_sha256": main_split.get("y_test_sha256"),
    }
    if any(direct_result.get(key) != value for key, value in direct_expected.items()):
        raise CampaignAggregationError(
            "Fresh MAIN direct validation has incompatible source hashes"
        )
    if (
        direct_result.get("evaluation_basis")
        != "direct_predictions_against_reproduced_test_labels"
        or direct_result.get("environment_truth_metadata_used") is not False
    ):
        raise CampaignAggregationError("Fresh MAIN direct validation basis is invalid")
    if not isinstance(direct_result.get("metrics"), Mapping) or not isinstance(
        direct_result.get("confusion_matrix"), Mapping
    ):
        raise CampaignAggregationError("Fresh MAIN direct validation metrics are invalid")
    _validate_metric_values(
        direct_result["metrics"],
        direct_result["confusion_matrix"],
        logical_id="main_direct_validation",
    )
    payloads["main_direct_validation"] = direct_result

    bootstrap = runs["main_bootstrap_ci"]
    _require_fresh_binding(bootstrap, main)
    bootstrap_result = _trusted_json(bootstrap, "bootstrap_ci")
    bootstrap_expected = {
        "source_run_id": source_run_id,
        "source_manifest_sha256": main.manifest_sha256,
        "source_predictions_sha256": source_hashes["predictions"],
        "n_boot": 10_000,
        "boot_seed": 12_345,
    }
    if any(bootstrap_result.get(key) != value for key, value in bootstrap_expected.items()):
        raise CampaignAggregationError("Fresh MAIN bootstrap provenance is incompatible")
    if (
        bootstrap.config.get("source_predictions_sha256") != source_hashes["predictions"]
        or bootstrap.config.get("n_resamples") != 10_000
        or bootstrap.config.get("bootstrap_seed") != 12_345
    ):
        raise CampaignAggregationError("Fresh MAIN bootstrap config is incompatible")
    payloads["main_bootstrap_ci"] = bootstrap_result

    duplicates = runs["main_duplicate_analysis"]
    _require_fresh_binding(duplicates, main)
    duplicate_result = _trusted_json(duplicates, "duplicate_analysis")
    duplicate_expected = {
        "source_run_id": source_run_id,
        "source_manifest_sha256": main.manifest_sha256,
        "source_cache_manifest_sha256": main_split.get("cache_manifest_sha256"),
        "source_dataset_sha256": main_split.get("source_csv_sha256"),
    }
    if any(duplicate_result.get(key) != value for key, value in duplicate_expected.items()):
        raise CampaignAggregationError("Fresh MAIN duplicate-analysis provenance is incompatible")
    verified_hashes = duplicate_result.get("verified_split_hashes")
    if not isinstance(verified_hashes, Mapping) or any(
        verified_hashes.get(key) != main_split.get(key) for key in _HASH_FIELDS
    ):
        raise CampaignAggregationError("Duplicate analysis uses an incompatible MAIN split")
    payloads["main_duplicate_analysis"] = duplicate_result

    shuffled = runs["shuffled_label_validation_s42_m42"]
    shuffled_metrics = _trusted_json(shuffled, "metrics")
    if (
        shuffled.config.get("timesteps") != 10_000
        or shuffled.config.get("job_classification") != "auxiliary_validation"
        or shuffled.config.get("counts_toward_primary_model_training_executions") is not False
        or shuffled.config.get("performance_comparison_eligible") is not False
        or shuffled_metrics.get("control_interpretation")
        != "anti_leakage_only_not_model_performance"
    ):
        raise CampaignAggregationError("Shuffled-label auxiliary control is incompatible")
    payloads["shuffled_label_validation_s42_m42"] = shuffled_metrics

    phase2 = runs["phase2_fresh_main"]
    _require_fresh_binding(phase2, main)
    configured_hashes = phase2.config.get("source_artifact_sha256")
    for key in ("model", "scaler", "train_percentiles", "feature_names", "manifest"):
        if not isinstance(configured_hashes, Mapping) or configured_hashes.get(key) != source_hashes[key]:
            raise CampaignAggregationError("Phase 2 source artifact hashes are incompatible")
    runtime = resolved_spec.get("resolved_runtime")
    expected_input_sha256 = (
        runtime.get("phase2_input_sha256") if isinstance(runtime, Mapping) else None
    )
    input_record = phase2.config.get("input")
    if (
        not _is_sha256(expected_input_sha256)
        or not isinstance(input_record, Mapping)
        or input_record.get("sha256") != expected_input_sha256
        or phase2.config.get("sensitive_metadata_exported") is not False
    ):
        raise CampaignAggregationError("Phase 2 input or privacy provenance is incompatible")
    payloads["phase2_metrics"] = _trusted_json(phase2, "metrics")
    payloads["phase2_diagnostics"] = _trusted_json(phase2, "diagnostics")
    return payloads


def _load_campaign_evidence(campaign_dir: Path) -> _CampaignEvidence:
    campaign_dir = campaign_dir.resolve()
    if not campaign_dir.is_dir():
        raise CampaignAggregationError(f"Campaign directory does not exist: {campaign_dir}")
    for filename in _REQUIRED_CAMPAIGN_FILES:
        if not (campaign_dir / filename).is_file():
            raise CampaignAggregationError(f"Campaign aggregation requires {filename}")

    original_spec = _read_json(
        campaign_dir / "campaign_spec_original.json",
        label="original campaign specification",
    )
    try:
        spec = validate_campaign_spec(original_spec)
    except CampaignSpecError as error:
        raise CampaignAggregationError(f"Campaign specification is incompatible: {error}") from error
    resolved_spec = _read_json(
        campaign_dir / "campaign_spec_resolved.json",
        label="resolved campaign specification",
    )
    if any(resolved_spec.get(key) != value for key, value in original_spec.items()):
        raise CampaignAggregationError("Resolved campaign specification changed locked fields")
    state = _read_json(campaign_dir / "campaign_state.json", label="campaign state")
    campaign_id = state.get("campaign_id")
    if (
        state.get("schema_version") != CAMPAIGN_STATE_SCHEMA_VERSION
        or not isinstance(campaign_id, str)
        or state.get("campaign_spec_sha256") != spec.content_hash
        or state.get("dispatch_mode") != "sequential"
    ):
        raise CampaignAggregationError("Campaign state identity is incompatible")
    state_entries = state.get("entries")
    if not isinstance(state_entries, Mapping) or set(state_entries) != set(spec.by_id):
        raise CampaignAggregationError("Campaign state entries differ from the locked specification")
    incomplete = sorted(
        logical_id
        for logical_id, record in state_entries.items()
        if not isinstance(record, Mapping)
        or record.get("status") not in {"completed", "reused"}
    )
    if incomplete:
        raise CampaignAggregationError(
            "Campaign is incomplete; aggregation blocked for: " + ", ".join(incomplete)
        )

    cache_sha256 = sha256_file(campaign_dir / "cache_manifest.json")
    preflight = _read_json(campaign_dir / "preflight_report.json", label="preflight report")
    runtime = resolved_spec.get("resolved_runtime")
    if (
        state.get("cache_manifest_sha256") != cache_sha256
        or preflight.get("report_sha256") != state.get("preflight_report_sha256")
        or not isinstance(runtime, Mapping)
        or runtime.get("cache_manifest_sha256") != cache_sha256
        or runtime.get("preflight_report_sha256") != state.get("preflight_report_sha256")
    ):
        raise CampaignAggregationError("Campaign cache/preflight binding is incompatible")

    runs: dict[str, _RunEvidence] = {}
    for entry in spec.entries:
        record = state_entries[entry.logical_id]
        if record.get("classification") != entry.classification or record.get("stage") != entry.stage:
            raise CampaignAggregationError(f"Campaign state metadata mismatch: {entry.logical_id}")
        snapshot = record.get("snapshot")
        if not isinstance(snapshot, Mapping) or snapshot.get("status") != "verified":
            raise CampaignAggregationError(
                f"Campaign is incomplete; snapshot is not verified: {entry.logical_id}"
            )
        if entry.classification != "alias":
            runs[entry.logical_id] = _load_physical_evidence(
                campaign_dir,
                campaign_id,
                entry,
                record,
            )

    for entry in spec.entries:
        if entry.classification != "alias":
            continue
        record = state_entries[entry.logical_id]
        source_id = entry.reuse_of
        if source_id is None or record.get("reuse_of") != source_id or source_id not in runs:
            raise CampaignAggregationError(f"Alias source is invalid: {entry.logical_id}")
        source = runs[source_id]
        _alias_dir, alias_relative = _safe_relative(
            campaign_dir,
            record.get("artifact_dir"),
            field=f"artifact_dir for alias {entry.logical_id}",
        )
        if (
            alias_relative != source.artifact_relative
            or record.get("source_manifest_sha256") != source.manifest_sha256
            or record.get("attempts") not in ([], None)
        ):
            raise CampaignAggregationError(f"Alias provenance is invalid: {entry.logical_id}")
        runs[entry.logical_id] = _RunEvidence(
            entry=entry,
            record=record,
            artifact_dir=source.artifact_dir,
            artifact_relative=source.artifact_relative,
            manifest=source.manifest,
            manifest_sha256=source.manifest_sha256,
            config=source.config,
            metrics=source.metrics,
            timing=source.timing,
            reuse_of=source_id,
            source=source,
        )

    auxiliary_payloads = _validate_auxiliary_payloads(runs, resolved_spec)
    return _CampaignEvidence(
        campaign_dir=campaign_dir,
        campaign_id=campaign_id,
        spec=spec,
        state=state,
        resolved_spec=resolved_spec,
        runs=runs,
        auxiliary_payloads=auxiliary_payloads,
    )


def _model_family(evidence: _RunEvidence) -> str:
    runner = evidence.physical.entry.runner
    if runner == "qrdqn":
        return "qrdqn"
    if runner == "random_forest":
        return "random_forest"
    return str(runner)


def _run_row(campaign: _CampaignEvidence, evidence: _RunEvidence) -> dict[str, Any]:
    physical = evidence.physical
    config = physical.config
    split = config.get("split_metadata")
    metrics = physical.metrics
    if not isinstance(split, Mapping) or not isinstance(metrics, Mapping):
        raise CampaignAggregationError(f"Cannot aggregate {evidence.entry.logical_id}")
    support = metrics.get("support")
    if not isinstance(support, Mapping):
        raise CampaignAggregationError(f"Cannot aggregate support for {evidence.entry.logical_id}")
    n_test = int(support["n_test"])
    return {
        "campaign_id": campaign.campaign_id,
        "logical_run_id": evidence.entry.logical_id,
        "physical_run_id": physical.manifest["run"]["physical_run_id"],
        "artifact_dir": physical.artifact_relative,
        "manifest_sha256": physical.manifest_sha256,
        "classification": evidence.entry.classification,
        "stage": evidence.entry.stage,
        "model_family": _model_family(evidence),
        "reuse_of": evidence.reuse_of,
        "source_logical_run_id": evidence.reuse_of,
        "split_mode": _resolved_config_value(config, "split_mode"),
        "split_seed": _resolved_config_value(config, "split_seed"),
        "model_seed": _resolved_config_value(config, "model_seed"),
        "timesteps": _resolved_config_value(config, "timesteps"),
        "train_max_rows": _resolved_config_value(config, "train_max_rows"),
        "train_rows": split.get("n_train"),
        "holdout_csv": _resolved_config_value(config, "holdout_csv"),
        "campaign_profile_id": campaign.spec.raw.get("profile_id"),
        "campaign_profile_hash": campaign.spec.raw.get("profile_hash"),
        "profile_id": config.get("profile_id"),
        "profile_hash": config.get("profile_hash"),
        **{key: split.get(key) for key in _HASH_FIELDS},
        "support": dict(support),
        "attack_prevalence": (
            None if n_test == 0 else float(int(support["attack"]) / n_test)
        ),
        "metrics": dict(metrics),
        "timings": dict(physical.timing or {}),
    }


def _csv_row(row: Mapping[str, Any]) -> dict[str, Any]:
    support = row.get("support")
    metrics = row.get("metrics")
    output = {field: row.get(field) for field in _CSV_FIELDS}
    if isinstance(support, Mapping):
        output.update(
            {
                "support_n_test": support.get("n_test"),
                "support_benign": support.get("benign"),
                "support_attack": support.get("attack"),
            }
        )
    if isinstance(metrics, Mapping):
        output.update({key: metrics.get(key) for key in (*_SUMMARY_METRICS, "tp", "tn", "fp", "fn")})
    output["timings_json"] = _canonical_json(row.get("timings", {}))
    return output


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                field: "" if row.get(field) is None else row.get(field)
                for field in fields
            }
        )
    atomic_write_text(path, stream.getvalue())


def _group(campaign: _CampaignEvidence, name: str, rows: Sequence[Mapping[str, Any]]) -> dict:
    return {
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "campaign_id": campaign.campaign_id,
        "group": name,
        "profile_id": campaign.spec.raw.get("profile_id"),
        "profile_hash": campaign.spec.raw.get("profile_hash"),
        "rows": list(rows),
    }


def _holdout_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    macro: dict[str, dict[str, Any]] = {}
    for metric in _SUMMARY_METRICS:
        defined = [
            float(row["metrics"][metric])
            for row in rows
            if isinstance(row.get("metrics"), Mapping)
            and isinstance(row["metrics"].get(metric), (int, float))
            and not isinstance(row["metrics"].get(metric), bool)
        ]
        macro[metric] = {
            "value": None if not defined else float(sum(defined) / len(defined)),
            "n_defined": len(defined),
        }
    pooled_counts = {
        key: sum(int(row["metrics"][key]) for row in rows)
        for key in ("tn", "fp", "fn", "tp")
    }
    pooled = confusion_to_metrics(
        pooled_counts["tn"],
        pooled_counts["fp"],
        pooled_counts["fn"],
        pooled_counts["tp"],
        undefined_metric_policy="null",
    )
    return {
        "n_folds": len(rows),
        "macro": macro,
        "pooled_confusion_matrix": pooled_counts,
        "pooled": pooled,
    }


def _comparison_rows(
    campaign: _CampaignEvidence,
    rows: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    pairs = [
        ("random_full", FRESH_MAIN_ID, "rf_random_full_s42_m42"),
        ("random_1m", "qrdqn_ladder_1m_s42_m42", "rf_random_1m_s42_m42"),
        ("day_full", "qrdqn_day_full_s42_m42", "rf_day_full_s42_m42"),
        ("holdout_webattacks", "qrdqn_holdout_webattacks_m42", "rf_holdout_webattacks_m42"),
        (
            "holdout_infilteration",
            "qrdqn_holdout_infilteration_m42",
            "rf_holdout_infilteration_m42",
        ),
        ("holdout_portscan", "qrdqn_holdout_portscan_m42", "rf_holdout_portscan_m42"),
        ("holdout_ddos", "qrdqn_holdout_ddos_m42", "rf_holdout_ddos_m42"),
    ]
    comparisons: list[dict[str, Any]] = []
    for comparison_id, qrdqn_id, rf_id in pairs:
        qrdqn = rows[qrdqn_id]
        rf = rows[rf_id]
        if (
            qrdqn.get("test_set_sha256") != rf.get("test_set_sha256")
            or qrdqn.get("y_test_sha256") != rf.get("y_test_sha256")
        ):
            raise CampaignAggregationError(
                f"Matched QRDQN/RF test partition is incompatible: {comparison_id}"
            )
        comparison = {
            "campaign_id": campaign.campaign_id,
            "comparison_id": comparison_id,
            "qrdqn_logical_run_id": qrdqn_id,
            "rf_logical_run_id": rf_id,
            "test_set_sha256": qrdqn["test_set_sha256"],
            "y_test_sha256": qrdqn["y_test_sha256"],
            "test_set_sha256_match": True,
            "qrdqn_manifest_sha256": qrdqn["manifest_sha256"],
            "rf_manifest_sha256": rf["manifest_sha256"],
        }
        for metric in _SUMMARY_METRICS:
            comparison[f"qrdqn_{metric}"] = qrdqn["metrics"].get(metric)
            comparison[f"rf_{metric}"] = rf["metrics"].get(metric)
        comparisons.append(comparison)
    return comparisons


def _auxiliary_document(
    campaign: _CampaignEvidence,
    logical_id: str,
    *,
    metric_scope: str,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    evidence = campaign.runs[logical_id]
    physical = evidence.physical
    return {
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "campaign_id": campaign.campaign_id,
        "group": logical_id,
        "metric_scope": metric_scope,
        "run": {
            "logical_run_id": logical_id,
            "physical_run_id": physical.manifest["run"]["physical_run_id"],
            "artifact_dir": physical.artifact_relative,
            "manifest_sha256": physical.manifest_sha256,
            "split_seed": physical.manifest["run"].get("split_seed"),
            "model_seed": physical.manifest["run"].get("model_seed"),
        },
        "config": dict(physical.config),
        "timings": dict(physical.timing or {}),
        "result": dict(result),
    }


def _aggregate_documents(campaign: _CampaignEvidence) -> dict[str, Any]:
    primary_rows = {
        entry.logical_id: _run_row(campaign, campaign.runs[entry.logical_id])
        for entry in campaign.spec.entries
        if entry.classification in {"primary_model_training", "alias"}
    }
    size_ids = [
        "qrdqn_ladder_100k_s42_m42",
        "qrdqn_ladder_250k_s42_m42",
        "qrdqn_ladder_500k_s42_m42",
        "qrdqn_ladder_1m_s42_m42",
        "qrdqn_ladder_2m_s42_m42",
        "qrdqn_ladder_full_s42_m42",
    ]
    seed_ids = ["qrdqn_seed_1m_s42_m42", *[f"qrdqn_seed_1m_s42_m{seed}" for seed in range(43, 47)]]
    holdout_ids = [
        *[
            f"qrdqn_holdout_{name}_m42"
            for name in ("webattacks", "infilteration", "portscan", "ddos")
        ],
        *[
            f"rf_holdout_{name}_m42"
            for name in ("webattacks", "infilteration", "portscan", "ddos")
        ],
    ]
    rf_ids = [
        entry.logical_id
        for entry in campaign.spec.entries
        if entry.runner == "random_forest"
    ]
    day_ids = ["qrdqn_day_full_s42_m42", "rf_day_full_s42_m42"]
    size_rows = [primary_rows[logical_id] for logical_id in size_ids]
    seed_rows = [primary_rows[logical_id] for logical_id in seed_ids]
    holdout_rows = [primary_rows[logical_id] for logical_id in holdout_ids]
    rf_rows = [primary_rows[logical_id] for logical_id in rf_ids]
    qrdqn_holdouts = [row for row in holdout_rows if row["model_family"] == "qrdqn"]
    rf_holdouts = [row for row in holdout_rows if row["model_family"] == "random_forest"]
    comparisons = _comparison_rows(campaign, primary_rows)

    targeted = _group(campaign, "targeted_holdouts", holdout_rows)
    targeted["summaries"] = {
        "qrdqn": _holdout_summary(qrdqn_holdouts),
        "random_forest": _holdout_summary(rf_holdouts),
    }
    phase2_document = _auxiliary_document(
        campaign,
        "phase2_fresh_main",
        metric_scope="phase2_offline_laboratory_domain",
        result=campaign.auxiliary_payloads["phase2_metrics"],
    )
    phase2_document["diagnostics"] = dict(
        campaign.auxiliary_payloads["phase2_diagnostics"]
    )
    comparison_fields = (
        "campaign_id",
        "comparison_id",
        "qrdqn_logical_run_id",
        "rf_logical_run_id",
        "test_set_sha256",
        "y_test_sha256",
        "test_set_sha256_match",
        "qrdqn_manifest_sha256",
        "rf_manifest_sha256",
        *[f"qrdqn_{metric}" for metric in _SUMMARY_METRICS],
        *[f"rf_{metric}" for metric in _SUMMARY_METRICS],
    )
    return {
        "main.json": _group(campaign, "main", [primary_rows[FRESH_MAIN_ID]]),
        "main_direct_validation.json": _auxiliary_document(
            campaign,
            "main_direct_validation",
            metric_scope="cicids2017_fresh_main_direct_validation",
            result=campaign.auxiliary_payloads["main_direct_validation"],
        ),
        "main_bootstrap_ci.json": _auxiliary_document(
            campaign,
            "main_bootstrap_ci",
            metric_scope="fixed_test_sampling_precision",
            result=campaign.auxiliary_payloads["main_bootstrap_ci"],
        ),
        "main_duplicate_analysis.json": _auxiliary_document(
            campaign,
            "main_duplicate_analysis",
            metric_scope="fresh_main_duplicate_and_cross_split_analysis",
            result=campaign.auxiliary_payloads["main_duplicate_analysis"],
        ),
        "shuffled_label_validation.json": _auxiliary_document(
            campaign,
            "shuffled_label_validation_s42_m42",
            metric_scope="auxiliary_anti_leakage_control",
            result=campaign.auxiliary_payloads["shuffled_label_validation_s42_m42"],
        ),
        "phase2_fresh_main.json": phase2_document,
        "day_split.json": _group(
            campaign,
            "day_split",
            [primary_rows[logical_id] for logical_id in day_ids],
        ),
        "size_ladder.json": _group(campaign, "size_ladder", size_rows),
        "size_ladder.csv": ([_csv_row(row) for row in size_rows], _CSV_FIELDS),
        "seed_sensitivity.json": _group(campaign, "seed_sensitivity", seed_rows),
        "seed_sensitivity.csv": ([_csv_row(row) for row in seed_rows], _CSV_FIELDS),
        "targeted_holdouts.json": targeted,
        "targeted_holdouts.csv": ([_csv_row(row) for row in holdout_rows], _CSV_FIELDS),
        "random_forest.json": _group(campaign, "random_forest", rf_rows),
        "random_forest.csv": ([_csv_row(row) for row in rf_rows], _CSV_FIELDS),
        "qrdqn_vs_rf.csv": (comparisons, comparison_fields),
    }


def _write_aggregate_directory(
    destination: Path,
    campaign: _CampaignEvidence,
    documents: Mapping[str, Any],
) -> None:
    for filename in AGGREGATE_DATA_FILES:
        document = documents[filename]
        if filename.endswith(".csv"):
            rows, fields = document
            _write_csv(destination / filename, rows, fields)
        else:
            atomic_write_json(destination / filename, document)
    inventory = {
        filename: {
            "sha256": sha256_file(destination / filename),
            "size_bytes": (destination / filename).stat().st_size,
        }
        for filename in AGGREGATE_DATA_FILES
    }
    summary = {
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "status": "completed",
        "campaign_complete": True,
        "campaign_id": campaign.campaign_id,
        "campaign_spec_sha256": campaign.spec.content_hash,
        "profile_id": campaign.spec.raw.get("profile_id"),
        "profile_hash": campaign.spec.raw.get("profile_hash"),
        "campaign_state_sha256": sha256_file(
            campaign.campaign_dir / "campaign_state.json"
        ),
        "cache_manifest_sha256": campaign.state.get("cache_manifest_sha256"),
        "preflight_report_sha256": campaign.state.get("preflight_report_sha256"),
        "counts": {
            "primary_physical_executions": campaign.spec.primary_execution_count,
            "primary_logical_result_points": campaign.spec.primary_logical_count,
            "auxiliary_jobs": campaign.spec.auxiliary_count,
            "aliases": campaign.spec.alias_count,
        },
        "aggregate_files": inventory,
    }
    atomic_write_json(destination / "campaign_summary.json", summary)


def aggregate_campaign(
    campaign_dir: Path | str,
    output_dir: Path | str,
    *,
    repo_root: Path | str = _REPO_ROOT,
) -> dict[str, Any]:
    """Validate a complete campaign and atomically write all aggregate groups."""
    campaign_dir = Path(campaign_dir)
    output_dir = _validate_output_path(
        Path(output_dir),
        repo_root=Path(repo_root),
        source_dir=campaign_dir,
    )
    campaign = _load_campaign_evidence(campaign_dir)
    documents = _aggregate_documents(campaign)
    _write_directory_atomically(
        output_dir,
        lambda temporary: _write_aggregate_directory(temporary, campaign, documents),
    )
    summary = validate_aggregate_directory(output_dir)
    return {
        "status": "completed",
        "campaign_id": campaign.campaign_id,
        "output_dir": str(output_dir),
        "files": list(AGGREGATE_FILES),
        "campaign_summary_sha256": sha256_file(output_dir / "campaign_summary.json"),
        "counts": summary["counts"],
    }


def validate_aggregate_directory(aggregate_dir: Path | str) -> dict[str, Any]:
    """Verify completeness and checksums before any renderer consumes aggregates."""
    aggregate_dir = Path(aggregate_dir).resolve()
    if not aggregate_dir.is_dir():
        raise CampaignAggregationError(
            f"Aggregate directory does not exist: {aggregate_dir}"
        )
    actual_files = {path.name for path in aggregate_dir.iterdir() if path.is_file()}
    if actual_files != set(AGGREGATE_FILES):
        raise CampaignAggregationError(
            "Aggregate directory is incomplete or contains undeclared files"
        )
    summary = _read_json(
        aggregate_dir / "campaign_summary.json",
        label="campaign aggregate summary",
    )
    inventory = summary.get("aggregate_files")
    if (
        summary.get("schema_version") != AGGREGATE_SCHEMA_VERSION
        or summary.get("status") != "completed"
        or summary.get("campaign_complete") is not True
        or not isinstance(inventory, Mapping)
        or set(inventory) != set(AGGREGATE_DATA_FILES)
    ):
        raise CampaignAggregationError("Campaign aggregate summary is incomplete")
    for filename, raw_record in inventory.items():
        if not isinstance(raw_record, Mapping):
            raise CampaignAggregationError(f"Invalid aggregate inventory: {filename}")
        path = aggregate_dir / filename
        if path.stat().st_size != raw_record.get("size_bytes"):
            raise CampaignAggregationError(f"Aggregate size/checksum mismatch: {filename}")
        if sha256_file(path) != raw_record.get("sha256"):
            raise CampaignAggregationError(f"Aggregate checksum mismatch: {filename}")
    return summary


def _metric(row: Mapping[str, Any], name: str = "f1_attack") -> float | None:
    metrics = row.get("metrics")
    value = metrics.get(name) if isinstance(metrics, Mapping) else None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _svg_chart(
    title: str,
    categories: Sequence[str],
    series: Sequence[tuple[str, Sequence[float | None]]],
) -> str:
    width, height = 1000, 560
    left, right, top, bottom = 80, 30, 60, 150
    plot_width = width - left - right
    plot_height = height - top - bottom
    count = max(1, len(categories))
    group_width = plot_width / count
    bar_width = max(3.0, min(44.0, group_width / max(1, len(series) + 1)))
    colors = ("#2563eb", "#dc2626", "#059669", "#7c3aed")
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="32" text-anchor="middle" '
        f'font-family="sans-serif" font-size="20">{escape(title)}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" '
        'stroke="#111827"/>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" '
        f'y2="{top + plot_height}" stroke="#111827"/>',
    ]
    for tick in range(6):
        value = tick / 5
        y = top + plot_height * (1 - value)
        parts.extend(
            [
                f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" '
                f'y2="{y:.2f}" stroke="#e5e7eb"/>',
                f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" '
                f'font-family="sans-serif" font-size="12">{value:.1f}</text>',
            ]
        )
    for index, category in enumerate(categories):
        center = left + group_width * (index + 0.5)
        total_width = bar_width * len(series)
        for series_index, (_name, values) in enumerate(series):
            value = values[index] if index < len(values) else None
            if value is None:
                continue
            bounded = max(0.0, min(1.0, float(value)))
            bar_height = bounded * plot_height
            x = center - total_width / 2 + series_index * bar_width
            y = top + plot_height - bar_height
            parts.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width - 2:.2f}" '
                f'height="{bar_height:.2f}" fill="{colors[series_index % len(colors)]}"/>'
            )
        parts.append(
            f'<text x="{center:.2f}" y="{top + plot_height + 18}" '
            'text-anchor="end" transform="rotate(-45 '
            f'{center:.2f} {top + plot_height + 18})" font-family="sans-serif" '
            f'font-size="11">{escape(str(category))}</text>'
        )
    for index, (name, _values) in enumerate(series):
        x = left + index * 180
        parts.extend(
            [
                f'<rect x="{x}" y="{height - 25}" width="14" height="14" '
                f'fill="{colors[index % len(colors)]}"/>',
                f'<text x="{x + 20}" y="{height - 13}" font-family="sans-serif" '
                f'font-size="12">{escape(name)}</text>',
            ]
        )
    parts.append("</svg>\n")
    return "".join(parts)


def _load_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except OSError as error:
        raise CampaignAggregationError(f"Cannot read aggregate CSV: {path}") from error


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise CampaignAggregationError(f"Invalid numeric aggregate value: {value!r}") from error


def _figure_documents(aggregate_dir: Path) -> dict[str, str]:
    size_rows = _read_json(aggregate_dir / "size_ladder.json")["rows"]
    seed_rows = _read_json(aggregate_dir / "seed_sensitivity.json")["rows"]
    day_rows = _read_json(aggregate_dir / "day_split.json")["rows"]
    holdout_rows = _read_json(aggregate_dir / "targeted_holdouts.json")["rows"]
    comparisons = _load_csv(aggregate_dir / "qrdqn_vs_rf.csv")
    return {
        "size_ladder.svg": _svg_chart(
            "QRDQN training-size ladder — F1 attack",
            [str(row.get("train_rows") or row.get("train_max_rows") or "full") for row in size_rows],
            [("QRDQN", [_metric(row) for row in size_rows])],
        ),
        "seed_sensitivity.svg": _svg_chart(
            "QRDQN seed sensitivity at fixed 1M-row budget — F1 attack",
            [str(row["model_seed"]) for row in seed_rows],
            [("QRDQN", [_metric(row) for row in seed_rows])],
        ),
        "day_generalisation.svg": _svg_chart(
            "Day-split generalisation — F1 attack",
            [str(row["model_family"]) for row in day_rows],
            [("F1 attack", [_metric(row) for row in day_rows])],
        ),
        "targeted_holdouts.svg": _svg_chart(
            "Targeted holdouts — F1 attack",
            [str(row["logical_run_id"]) for row in holdout_rows],
            [("F1 attack", [_metric(row) for row in holdout_rows])],
        ),
        "qrdqn_vs_rf.svg": _svg_chart(
            "Matched QRDQN vs Random Forest — F1 attack",
            [row["comparison_id"] for row in comparisons],
            [
                ("QRDQN", [_float_or_none(row.get("qrdqn_f1_attack")) for row in comparisons]),
                ("Random Forest", [_float_or_none(row.get("rf_f1_attack")) for row in comparisons]),
            ],
        ),
    }


def _write_figure_directory(
    destination: Path,
    aggregate_dir: Path,
    summary: Mapping[str, Any],
    figures: Mapping[str, str],
) -> None:
    for filename in FIGURE_FILES:
        atomic_write_text(destination / filename, figures[filename])
    inventory = {
        filename: {
            "sha256": sha256_file(destination / filename),
            "size_bytes": (destination / filename).stat().st_size,
        }
        for filename in FIGURE_FILES
    }
    atomic_write_json(
        destination / "figure_manifest.json",
        {
            "schema_version": FIGURE_SCHEMA_VERSION,
            "status": "completed",
            "source_campaign_id": summary.get("campaign_id"),
            "source_campaign_summary_sha256": sha256_file(
                aggregate_dir / "campaign_summary.json"
            ),
            "metric": "f1_attack",
            "figures": inventory,
        },
    )


def generate_campaign_figures(
    aggregate_dir: Path | str,
    output_dir: Path | str,
    *,
    repo_root: Path | str = _REPO_ROOT,
) -> dict[str, Any]:
    """Render deterministic future SVGs only from a verified complete aggregate."""
    aggregate_dir = Path(aggregate_dir).resolve()
    summary = validate_aggregate_directory(aggregate_dir)
    output_dir = _validate_output_path(
        Path(output_dir),
        repo_root=Path(repo_root),
        source_dir=aggregate_dir,
    )
    figures = _figure_documents(aggregate_dir)
    _write_directory_atomically(
        output_dir,
        lambda temporary: _write_figure_directory(
            temporary,
            aggregate_dir,
            summary,
            figures,
        ),
    )
    return {
        "status": "completed",
        "source_campaign_id": summary.get("campaign_id"),
        "output_dir": str(output_dir),
        "files": ["figure_manifest.json", *FIGURE_FILES],
    }


__all__ = [
    "AGGREGATE_FILES",
    "AGGREGATE_SCHEMA_VERSION",
    "CampaignAggregationError",
    "FIGURE_FILES",
    "FIGURE_SCHEMA_VERSION",
    "aggregate_campaign",
    "generate_campaign_figures",
    "validate_aggregate_directory",
]
