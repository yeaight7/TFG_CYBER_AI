from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.artifact_integrity import artifact_manifest_entry, sha256_file
from src.campaign import FRESH_MAIN_ID, load_campaign_spec
from src.campaign_aggregation import (
    CampaignAggregationError,
    aggregate_campaign,
    generate_campaign_figures,
    validate_aggregate_directory,
)
from src.experiment_profiles import MAIN_V1_PROFILE_HASH
from src.metrics_utils import confusion_to_metrics
from src.run_artifacts import (
    ArtifactManifestWriter,
    ArtifactRequirement,
    atomic_write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = REPO_ROOT / "experiments" / "final_experiment_campaign.json"
CAMPAIGN_ID = "synthetic-phase-8"
EXPECTED_AGGREGATES = {
    "campaign_summary.json",
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
}
EXPECTED_FIGURES = {
    "figure_manifest.json",
    "size_ladder.svg",
    "seed_sensitivity.svg",
    "day_generalisation.svg",
    "targeted_holdouts.svg",
    "qrdqn_vs_rf.svg",
}


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _partition_hashes(entry) -> tuple[str, str]:
    if entry.config.get("split_mode") == "day":
        key = "day"
    elif entry.config.get("split_mode") == "exact-holdout":
        key = str(entry.config["holdout_csv"])
    else:
        key = "random-s42"
    return _digest(f"test:{key}"), _digest(f"labels:{key}")


def _confusion(entry) -> tuple[int, int, int, int]:
    if entry.logical_id == "qrdqn_holdout_webattacks_m42":
        return 80, 0, 20, 0
    offset = int(_digest(entry.logical_id)[:2], 16) % 4
    return 80 - offset, 2 + offset, 3, 15


def _metric_payload(entry) -> dict:
    tn, fp, fn, tp = _confusion(entry)
    metrics = confusion_to_metrics(
        tn,
        fp,
        fn,
        tp,
        undefined_metric_policy="null",
    )
    metrics.update(
        {
            "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
            "support": {
                "n_test": tn + fp + fn + tp,
                "benign": tn + fp,
                "attack": fn + tp,
            },
        }
    )
    return metrics


def _write_primary_attempt(
    campaign_dir: Path,
    entry,
    *,
    cache_manifest_sha256: str,
    mismatched_rf_pair: bool,
) -> dict:
    attempt_dir = campaign_dir / "attempts" / entry.logical_id / "attempt-1"
    test_hash, y_test_hash = _partition_hashes(entry)
    if mismatched_rf_pair and entry.logical_id == "rf_day_full_s42_m42":
        test_hash = _digest("mismatched-rf-day-test")
    split_metadata = {
        "split_mode": entry.config["split_mode"],
        "split_seed": entry.config["split_seed"],
        "train_set_sha256": _digest(f"train:{entry.logical_id}"),
        "y_train_sha256": _digest(f"train-labels:{entry.logical_id}"),
        "test_set_sha256": test_hash,
        "y_test_sha256": y_test_hash,
        "cache_manifest_sha256": cache_manifest_sha256,
        "source_csv_sha256": {"synthetic.csv": _digest("synthetic.csv")},
        "n_train": entry.config.get("train_max_rows") or 2_000_000,
        "n_test": 100,
    }
    request = {
        "split_mode": entry.config["split_mode"],
        "split_seed": entry.config["split_seed"],
        "model_seed": entry.config["model_seed"],
        "train_max_rows": entry.config.get("train_max_rows"),
        "holdout_csv": entry.config.get("holdout_csv"),
    }
    if entry.runner == "qrdqn":
        request.update({"timesteps": entry.config["timesteps"], "profile_id": "main-v1"})
    config = {
        "status": "completed",
        "run_id": "attempt-1",
        "algorithm": "QRDQN" if entry.runner == "qrdqn" else "RandomForest",
        "request": request,
        "split_mode": entry.config["split_mode"],
        "split_seed": entry.config["split_seed"],
        "model_seed": entry.config["model_seed"],
        "train_max_rows": entry.config.get("train_max_rows"),
        "holdout_csv": entry.config.get("holdout_csv"),
        "split_metadata": split_metadata,
    }
    run_metadata = {
        "campaign_id": CAMPAIGN_ID,
        "logical_run_id": entry.logical_id,
        "physical_run_id": "attempt-1",
        "attempt": 1,
        "split_seed": entry.config["split_seed"],
        "model_seed": entry.config["model_seed"],
    }
    requirements = {
        "config": ArtifactRequirement("config.json"),
        "metrics": ArtifactRequirement("metrics.json"),
        "timing": ArtifactRequirement("timing.json"),
    }
    if entry.runner == "qrdqn":
        config.update(
            {
                "profile_id": "main-v1",
                "profile_hash": MAIN_V1_PROFILE_HASH,
                "timesteps": entry.config["timesteps"],
            }
        )
        run_metadata.update(
            {"profile_id": "main-v1", "profile_hash": MAIN_V1_PROFILE_HASH}
        )
    else:
        run_metadata["algorithm"] = "RandomForest"
    if entry.logical_id == FRESH_MAIN_ID:
        requirements.update(
            {
                "model": ArtifactRequirement("model.zip"),
                "scaler": ArtifactRequirement("scaler.joblib"),
                "predictions": ArtifactRequirement("predictions.npz"),
                "train_percentiles": ArtifactRequirement("train_percentiles.npz"),
                "feature_names": ArtifactRequirement("feature_names.json"),
            }
        )

    writer = ArtifactManifestWriter(
        attempt_dir,
        run_metadata=run_metadata,
        requirements=requirements,
    )
    writer.start()
    atomic_write_json(attempt_dir / "config.json", config)
    atomic_write_json(attempt_dir / "metrics.json", _metric_payload(entry))
    atomic_write_json(
        attempt_dir / "timing.json",
        {
            "phases": {
                "training": {
                    "duration_seconds": 1.0,
                    "units": entry.config.get("timesteps") or split_metadata["n_train"],
                    "unit": "timesteps" if entry.runner == "qrdqn" else "rows",
                    "throughput_per_second": 100.0,
                },
                "evaluation": {
                    "duration_seconds": 0.25,
                    "units": 100,
                    "unit": "rows",
                    "throughput_per_second": 400.0,
                },
            }
        },
    )
    if entry.logical_id == FRESH_MAIN_ID:
        (attempt_dir / "model.zip").write_bytes(b"synthetic-main-model")
        (attempt_dir / "scaler.joblib").write_bytes(b"synthetic-main-scaler")
        np.savez_compressed(
            attempt_dir / "predictions.npz",
            y_true=np.array([0, 0, 1, 1], dtype=np.int64),
            y_pred=np.array([0, 0, 1, 1], dtype=np.int64),
        )
        np.savez_compressed(
            attempt_dir / "train_percentiles.npz",
            p_low=np.zeros(76),
            p_high=np.ones(76),
        )
        atomic_write_json(
            attempt_dir / "feature_names.json",
            [f"feature_{index}" for index in range(152)],
        )
    return writer.complete()

def _fresh_source(main_dir: Path) -> dict[str, str]:
    manifest = _read_json(main_dir / "artifact_manifest.json")
    return {
        "run_id": "attempt-1",
        "manifest": sha256_file(main_dir / "artifact_manifest.json"),
        "model": str(artifact_manifest_entry(manifest, "model")["sha256"]),
        "scaler": str(artifact_manifest_entry(manifest, "scaler")["sha256"]),
        "predictions": str(artifact_manifest_entry(manifest, "predictions")["sha256"]),
        "train_percentiles": str(
            artifact_manifest_entry(manifest, "train_percentiles")["sha256"]
        ),
        "feature_names": str(
            artifact_manifest_entry(manifest, "feature_names")["sha256"]
        ),
    }


def _write_auxiliary_attempt(
    campaign_dir: Path,
    entry,
    *,
    main_source: dict[str, str],
    main_config: dict,
    phase2_input_sha256: str,
    historical_auxiliary: bool,
) -> dict:
    attempt_dir = campaign_dir / "attempts" / entry.logical_id / "attempt-1"
    source_manifest = main_source["manifest"]
    if historical_auxiliary and entry.logical_id == "main_direct_validation":
        source_manifest = _digest("historical-main")
    is_fresh_derived = entry.logical_id != "shuffled_label_validation_s42_m42"
    run_metadata = {
        "campaign_id": CAMPAIGN_ID,
        "logical_run_id": entry.logical_id,
        "physical_run_id": "attempt-1",
        "attempt": 1,
        "split_seed": 42,
        "model_seed": 42,
    }
    if is_fresh_derived:
        run_metadata.update(
            {
                "source_run_id": main_source["run_id"],
                "source_manifest_sha256": source_manifest,
            }
        )
    config = {
        "job_type": entry.runner,
        "source_run_id": main_source["run_id"] if is_fresh_derived else None,
        "source_manifest_sha256": source_manifest if is_fresh_derived else None,
        "split_seed": 42,
        "model_seed": 42,
    }
    requirements = {"config": ArtifactRequirement("config.json")}
    payloads: dict[str, dict] = {}

    if entry.logical_id == "main_direct_validation":
        main_metrics = _read_json(
            campaign_dir
            / "attempts"
            / FRESH_MAIN_ID
            / "attempt-1"
            / "metrics.json"
        )
        requirements["validation_results"] = ArtifactRequirement("validation_results.json")
        payloads["validation_results.json"] = {
            "source_run_id": main_source["run_id"],
            "source_manifest_sha256": source_manifest,
            "source_model_sha256": main_source["model"],
            "source_scaler_sha256": main_source["scaler"],
            "test_set_sha256": main_config["split_metadata"]["test_set_sha256"],
            "y_test_sha256": main_config["split_metadata"]["y_test_sha256"],
            "evaluation_basis": "direct_predictions_against_reproduced_test_labels",
            "environment_truth_metadata_used": False,
            "confusion_matrix": main_metrics["confusion_matrix"],
            "metrics": {
                key: value
                for key, value in main_metrics.items()
                if key not in {"confusion_matrix", "support"}
            },
        }
    elif entry.logical_id == "main_bootstrap_ci":
        requirements["bootstrap_ci"] = ArtifactRequirement("bootstrap_ci.json")
        payloads["bootstrap_ci.json"] = {
            "source_run_id": main_source["run_id"],
            "source_manifest_sha256": source_manifest,
            "source_predictions_sha256": main_source["predictions"],
            "n_test": 100,
            "n_boot": 10_000,
            "boot_seed": 12_345,
            "bootstrap": "stratified",
            "ci_level": 0.95,
            "confusion_counts": {"tn": 80, "fp": 2, "fn": 3, "tp": 15},
            "metrics_ci": {
                "f1_attack": {
                    "point": 0.857143,
                    "boot_mean": 0.85,
                    "boot_std": 0.01,
                    "ci95_low": 0.82,
                    "ci95_high": 0.89,
                }
            },
        }
        config.update(
            {
                "source_predictions_sha256": main_source["predictions"],
                "bootstrap_seed": 12_345,
                "n_resamples": 10_000,
            }
        )
    elif entry.logical_id == "main_duplicate_analysis":
        requirements["duplicate_analysis"] = ArtifactRequirement("duplicate_analysis.json")
        split_hashes = {
            key: main_config["split_metadata"][key]
            for key in (
                "train_set_sha256",
                "y_train_sha256",
                "test_set_sha256",
                "y_test_sha256",
            )
        }
        payloads["duplicate_analysis.json"] = {
            "job_type": "fresh_main_duplicate_cross_split_analysis",
            "source_run_id": main_source["run_id"],
            "source_manifest_sha256": source_manifest,
            "source_cache_manifest_sha256": main_config["split_metadata"][
                "cache_manifest_sha256"
            ],
            "source_dataset_sha256": main_config["split_metadata"]["source_csv_sha256"],
            "verified_split_hashes": split_hashes,
            "counts": {"n_train": 2_000_000, "n_test": 100, "n_total": 2_000_100},
            "cross_split_leakage_feature_only": {
                "n_test_rows_also_in_train": 2,
                "pct_of_test": 2.0,
            },
        }
        config["source_cache_manifest_sha256"] = main_config["split_metadata"][
            "cache_manifest_sha256"
        ]
    elif entry.logical_id == "shuffled_label_validation_s42_m42":
        requirements.update(
            {
                "metrics": ArtifactRequirement("metrics.json"),
                "timing": ArtifactRequirement("timing.json"),
            }
        )
        config = {
            "job_type": "shuffled_label_validation",
            "status": "completed",
            "split_seed": 42,
            "model_seed": 42,
            "timesteps": 10_000,
            "job_classification": "auxiliary_validation",
            "counts_toward_primary_model_training_executions": False,
            "performance_comparison_eligible": False,
            "label_permutation_sha256": _digest("label-permutation"),
        }
        payloads["metrics.json"] = {
            "accuracy": 0.5,
            "shuffled_accuracy": 0.5,
            "baseline_accuracy": 0.6,
            "leakage_detected": False,
            "control_interpretation": "anti_leakage_only_not_model_performance",
            "confusion_matrix": {"tn": 50, "fp": 10, "fn": 40, "tp": 0},
        }
        payloads["timing.json"] = {
            "phases": {"training": {"duration_seconds": 0.5}}
        }
    elif entry.logical_id == "phase2_fresh_main":
        requirements.update(
            {
                "metrics": ArtifactRequirement("metrics.json"),
                "diagnostics": ArtifactRequirement("diagnostics.json"),
                "timing": ArtifactRequirement("timing.json"),
            }
        )
        source_hashes = {
            "model": main_source["model"],
            "scaler": main_source["scaler"],
            "train_percentiles": main_source["train_percentiles"],
            "feature_names": main_source["feature_names"],
            "manifest": main_source["manifest"],
        }
        config.update(
            {
                "source_artifact_sha256": source_hashes,
                "input": {
                    "filename": "synthetic-lab-flows.csv",
                    "size_bytes": 123,
                    "sha256": phase2_input_sha256,
                },
                "preprocessing": {
                    "percentile_clipping": "training_p0.5_p99.5",
                    "clip_z": 10.0,
                    "scaler": "fresh_main_train_only_standard_scaler",
                },
                "sensitive_metadata_exported": False,
                "truth_labels_available": True,
            }
        )
        payloads["metrics.json"] = {
            "n_flows": 50,
            "block_rate": 0.2,
            "allow_rate": 0.8,
            "accuracy": 0.9,
        }
        payloads["diagnostics.json"] = {"z_abs_max": 4.0, "z_abs_mean": 0.5}
        payloads["timing.json"] = {
            "phases": {"inference": {"duration_seconds": 0.1}}
        }
    else:  # pragma: no cover - locked spec makes this unreachable
        raise AssertionError(entry.logical_id)

    writer = ArtifactManifestWriter(
        attempt_dir,
        run_metadata=run_metadata,
        requirements=requirements,
    )
    writer.start()
    atomic_write_json(attempt_dir / "config.json", config)
    for filename, payload in payloads.items():
        atomic_write_json(attempt_dir / filename, payload)
    return writer.complete()


def _synthetic_campaign(
    tmp_path: Path,
    *,
    complete: bool = True,
    mismatched_rf_pair: bool = False,
    historical_auxiliary: bool = False,
) -> Path:
    spec = load_campaign_spec(SPEC_PATH)
    campaign_dir = tmp_path / "campaign"
    campaign_dir.mkdir(parents=True)
    cache_manifest = campaign_dir / "cache_manifest.json"
    atomic_write_json(cache_manifest, {"validation_status": "valid"})
    cache_manifest_sha256 = sha256_file(cache_manifest)
    phase2_input_sha256 = _digest("synthetic-lab-input")
    preflight_report_sha256 = _digest("synthetic-preflight")
    atomic_write_json(
        campaign_dir / "preflight_report.json",
        {"status": "passed", "report_sha256": preflight_report_sha256},
    )
    atomic_write_json(campaign_dir / "campaign_spec_original.json", spec.raw)
    atomic_write_json(
        campaign_dir / "campaign_spec_resolved.json",
        {
            **dict(spec.raw),
            "resolved_runtime": {
                "cache_manifest_sha256": cache_manifest_sha256,
                "preflight_report_sha256": preflight_report_sha256,
                "phase2_input_sha256": phase2_input_sha256,
            },
        },
    )

    manifests: dict[str, dict] = {}
    for entry in spec.entries:
        if entry.classification == "primary_model_training":
            manifests[entry.logical_id] = _write_primary_attempt(
                campaign_dir,
                entry,
                cache_manifest_sha256=cache_manifest_sha256,
                mismatched_rf_pair=mismatched_rf_pair,
            )

    main_dir = campaign_dir / "attempts" / FRESH_MAIN_ID / "attempt-1"
    main_source = _fresh_source(main_dir)
    main_config = _read_json(main_dir / "config.json")
    for entry in spec.entries:
        if entry.classification == "auxiliary":
            manifests[entry.logical_id] = _write_auxiliary_attempt(
                campaign_dir,
                entry,
                main_source=main_source,
                main_config=main_config,
                phase2_input_sha256=phase2_input_sha256,
                historical_auxiliary=historical_auxiliary,
            )

    state_entries = {}
    for entry in spec.entries:
        if entry.classification == "alias":
            source_id = str(entry.reuse_of)
            source_dir = campaign_dir / "attempts" / source_id / "attempt-1"
            state_entries[entry.logical_id] = {
                "classification": "alias",
                "stage": entry.stage,
                "status": "reused",
                "attempts": [],
                "reuse_of": source_id,
                "source_manifest_sha256": sha256_file(
                    source_dir / "artifact_manifest.json"
                ),
                "artifact_dir": f"attempts/{source_id}/attempt-1",
            }
            continue
        attempt_dir = campaign_dir / "attempts" / entry.logical_id / "attempt-1"
        state_entries[entry.logical_id] = {
            "classification": entry.classification,
            "stage": entry.stage,
            "status": "completed",
            "attempts": [
                {
                    "attempt": 1,
                    "physical_run_id": "attempt-1",
                    "artifact_dir": f"attempts/{entry.logical_id}/attempt-1",
                    "status": "completed",
                    "manifest_sha256": sha256_file(
                        attempt_dir / "artifact_manifest.json"
                    ),
                    "verified_schema_version": "3.0",
                }
            ],
            "export": {"status": "verified"},
        }

    if not complete:
        pending = state_entries["rf_holdout_ddos_m42"]
        pending["status"] = "pending"
        pending["attempts"] = []
        pending.pop("export", None)

    atomic_write_json(
        campaign_dir / "campaign_state.json",
        {
            "schema_version": "1.0",
            "campaign_id": CAMPAIGN_ID,
            "campaign_spec_id": "final-experiment-v1",
            "campaign_spec_sha256": spec.content_hash,
            "dispatch_mode": "sequential",
            "cache_manifest_sha256": cache_manifest_sha256,
            "preflight_report_sha256": preflight_report_sha256,
            "entries": state_entries,
        },
    )
    return campaign_dir


def test_aggregation_writes_exact_groups_macros_and_alias_provenance(tmp_path: Path):
    campaign_dir = _synthetic_campaign(tmp_path)
    output_dir = tmp_path / "aggregates"

    report = aggregate_campaign(campaign_dir, output_dir, repo_root=tmp_path)

    assert report["status"] == "completed"
    assert {path.name for path in output_dir.iterdir()} == EXPECTED_AGGREGATES
    summary = validate_aggregate_directory(output_dir)
    assert summary["campaign_complete"] is True
    assert summary["counts"] == {
        "primary_physical_executions": 22,
        "primary_logical_result_points": 24,
        "auxiliary_jobs": 5,
        "aliases": 2,
    }
    assert summary["profile_id"] == "main-v1"
    assert summary["profile_hash"] == MAIN_V1_PROFILE_HASH

    main_row = _read_json(output_dir / "main.json")["rows"][0]
    assert main_row["campaign_profile_id"] == "main-v1"
    assert main_row["campaign_profile_hash"] == MAIN_V1_PROFILE_HASH
    assert main_row["support"] == {"n_test": 100, "benign": 82, "attack": 18}
    assert main_row["timings"]["phases"]["training"]["duration_seconds"] == 1.0

    size_rows = _read_json(output_dir / "size_ladder.json")["rows"]
    assert len(size_rows) == 6
    full = next(row for row in size_rows if row["logical_run_id"].endswith("full_s42_m42"))
    assert full["reuse_of"] == FRESH_MAIN_ID
    assert full["artifact_dir"] == f"attempts/{FRESH_MAIN_ID}/attempt-1"
    assert len({row["logical_run_id"] for row in size_rows}) == 6

    seed_rows = _read_json(output_dir / "seed_sensitivity.json")["rows"]
    assert [row["model_seed"] for row in seed_rows] == [42, 43, 44, 45, 46]
    assert seed_rows[0]["reuse_of"] == "qrdqn_ladder_1m_s42_m42"
    assert len(_read_json(output_dir / "random_forest.json")["rows"]) == 7

    holdouts = _read_json(output_dir / "targeted_holdouts.json")
    assert len(holdouts["rows"]) == 8
    qrdqn_summary = holdouts["summaries"]["qrdqn"]
    assert qrdqn_summary["macro"]["precision_attack"]["n_defined"] == 3
    pooled_counts = qrdqn_summary["pooled_confusion_matrix"]
    assert qrdqn_summary["pooled"]["tp"] == pooled_counts["tp"]
    assert qrdqn_summary["pooled"]["tn"] == pooled_counts["tn"]

    with (output_dir / "targeted_holdouts.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        csv_rows = list(csv.DictReader(handle))
    undefined = next(
        row for row in csv_rows if row["logical_run_id"] == "qrdqn_holdout_webattacks_m42"
    )
    assert undefined["precision_attack"] == ""

    comparisons = list(
        csv.DictReader((output_dir / "qrdqn_vs_rf.csv").open(encoding="utf-8", newline=""))
    )
    assert len(comparisons) == 7
    assert all(row["test_set_sha256_match"] == "True" for row in comparisons)

    shuffled = _read_json(output_dir / "shuffled_label_validation.json")
    assert shuffled["metric_scope"] == "auxiliary_anti_leakage_control"
    phase2 = _read_json(output_dir / "phase2_fresh_main.json")
    assert phase2["metric_scope"] == "phase2_offline_laboratory_domain"
    for filename in (
        "main.json",
        "day_split.json",
        "size_ladder.json",
        "seed_sensitivity.json",
        "targeted_holdouts.json",
        "random_forest.json",
    ):
        assert "shuffled_label_validation_s42_m42" not in (
            output_dir / filename
        ).read_text(encoding="utf-8")
        assert "phase2_fresh_main" not in (output_dir / filename).read_text(
            encoding="utf-8"
        )


def test_aggregation_refuses_incomplete_campaign_without_partial_output(tmp_path: Path):
    campaign_dir = _synthetic_campaign(tmp_path, complete=False)
    output_dir = tmp_path / "aggregates"

    with pytest.raises(CampaignAggregationError, match="incomplete"):
        aggregate_campaign(campaign_dir, output_dir, repo_root=tmp_path)

    assert not output_dir.exists()


def test_aggregation_does_not_treat_external_export_as_durability_gate(
    tmp_path: Path,
) -> None:
    campaign_dir = _synthetic_campaign(tmp_path, complete=True)
    state_path = campaign_dir / "campaign_state.json"
    state = _read_json(state_path)
    state["entries"][FRESH_MAIN_ID]["export"] = {
        "status": "failed",
        "error": "operator will retry or download separately",
    }
    atomic_write_json(state_path, state)

    output_dir = tmp_path / "aggregates"
    result = aggregate_campaign(campaign_dir, output_dir, repo_root=tmp_path)

    assert result["status"] == "completed"


def test_aggregation_rejects_incompatible_matched_partition(tmp_path: Path):
    campaign_dir = _synthetic_campaign(tmp_path, mismatched_rf_pair=True)

    with pytest.raises(CampaignAggregationError, match="test partition"):
        aggregate_campaign(campaign_dir, tmp_path / "aggregates", repo_root=tmp_path)


def test_aggregation_rejects_historical_auxiliary_substitute(tmp_path: Path):
    campaign_dir = _synthetic_campaign(tmp_path, historical_auxiliary=True)

    with pytest.raises(CampaignAggregationError, match="fresh campaign MAIN"):
        aggregate_campaign(campaign_dir, tmp_path / "aggregates", repo_root=tmp_path)


@pytest.mark.parametrize("protected", ["memoria", "report"])
def test_aggregation_and_figures_refuse_protected_output_paths(
    tmp_path: Path,
    protected: str,
):
    campaign_dir = _synthetic_campaign(tmp_path / "source")
    protected_output = tmp_path / protected / "campaign"

    with pytest.raises(CampaignAggregationError, match="protected"):
        aggregate_campaign(campaign_dir, protected_output, repo_root=tmp_path)

    aggregate_dir = tmp_path / "safe-aggregates"
    aggregate_campaign(campaign_dir, aggregate_dir, repo_root=tmp_path)
    with pytest.raises(CampaignAggregationError, match="protected"):
        generate_campaign_figures(
            aggregate_dir,
            tmp_path / protected / "figures",
            repo_root=tmp_path,
        )


def test_figure_generator_requires_validated_aggregates_and_writes_future_svgs(
    tmp_path: Path,
):
    campaign_dir = _synthetic_campaign(tmp_path)
    aggregate_dir = tmp_path / "aggregates"
    figure_dir = tmp_path / "figures"
    aggregate_campaign(campaign_dir, aggregate_dir, repo_root=tmp_path)

    report = generate_campaign_figures(aggregate_dir, figure_dir, repo_root=tmp_path)

    assert report["status"] == "completed"
    assert {path.name for path in figure_dir.iterdir()} == EXPECTED_FIGURES
    manifest = _read_json(figure_dir / "figure_manifest.json")
    assert manifest["source_campaign_id"] == CAMPAIGN_ID
    assert manifest["source_campaign_summary_sha256"] == sha256_file(
        aggregate_dir / "campaign_summary.json"
    )
    assert all((figure_dir / name).read_text(encoding="utf-8").startswith("<svg") for name in EXPECTED_FIGURES if name.endswith(".svg"))

    (aggregate_dir / "size_ladder.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(CampaignAggregationError, match="checksum"):
        generate_campaign_figures(
            aggregate_dir,
            tmp_path / "invalid-figures",
            repo_root=tmp_path,
        )
    assert not (tmp_path / "invalid-figures").exists()


def test_phase8_clis_aggregate_and_render_from_explicit_paths(tmp_path: Path, capsys):
    from scripts.aggregate_campaign import main as aggregate_main
    from scripts.generate_campaign_figures import main as figure_main

    campaign_dir = _synthetic_campaign(tmp_path)
    aggregate_dir = tmp_path / "cli-aggregates"
    figure_dir = tmp_path / "cli-figures"

    assert (
        aggregate_main(
            [
                "--campaign-dir",
                str(campaign_dir),
                "--output-dir",
                str(aggregate_dir),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "completed"

    assert (
        figure_main(
            [
                "--aggregate-dir",
                str(aggregate_dir),
                "--output-dir",
                str(figure_dir),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "completed"
