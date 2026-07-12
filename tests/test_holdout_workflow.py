from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.artifact_integrity import verify_artifact_manifest
from src.qrdqn_experiment import PreparedSplit, run_qrdqn_experiment
from src.validate_leave_one_csv_out import (
    TARGETED_QRDQN_HOLDOUTS,
    TargetedHoldoutWorkflowConfig,
    aggregate_holdout_metrics,
    resolve_targeted_holdouts,
    run_targeted_holdouts,
)


EXPECTED_HOLDOUTS = {
    "qrdqn_holdout_webattacks_m42":
        "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "qrdqn_holdout_infilteration_m42":
        "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "qrdqn_holdout_portscan_m42":
        "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "qrdqn_holdout_ddos_m42":
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
}


def test_targeted_holdout_list_is_exact_and_default_never_expands_to_all_eight():
    available = ["Monday-WorkingHours.pcap_ISCX.csv", *EXPECTED_HOLDOUTS.values()]

    assert TARGETED_QRDQN_HOLDOUTS == EXPECTED_HOLDOUTS
    assert resolve_targeted_holdouts(None, available) == list(EXPECTED_HOLDOUTS.values())


def test_targeted_holdout_selection_rejects_unknown_duplicates_and_non_exact_names():
    available = list(EXPECTED_HOLDOUTS.values())
    selected = available[0]

    with pytest.raises(ValueError, match="targeted holdout"):
        resolve_targeted_holdouts(["Wednesday-workingHours.pcap_ISCX.csv"], available)
    with pytest.raises(ValueError, match="duplicate"):
        resolve_targeted_holdouts([selected, selected], available)
    with pytest.raises(ValueError, match="exact"):
        resolve_targeted_holdouts([selected.lower()], available)


def test_locked_targeted_workflow_rejects_non_campaign_model_seed(tmp_path: Path):
    with pytest.raises(ValueError, match="model_seed=42"):
        TargetedHoldoutWorkflowConfig(
            artifact_root=tmp_path / "runs",
            dataset_root=tmp_path / "dataset",
            cache_root=tmp_path / "cache",
            model_seed=43,
        )


def test_holdout_aggregation_keeps_nulls_out_of_macro_and_pools_confusion_counts():
    aggregate = aggregate_holdout_metrics(
        [
            {
                "accuracy": 0.8,
                "precision_attack": None,
                "confusion_matrix": {"tn": 8, "fp": 2, "fn": 0, "tp": 0},
            },
            {
                "accuracy": 0.5,
                "precision_attack": 0.5,
                "confusion_matrix": {"tn": 4, "fp": 1, "fn": 4, "tp": 1},
            },
        ]
    )

    assert aggregate["defined_only_macro"]["accuracy"] == {
        "mean": pytest.approx(0.65),
        "n_defined": 2,
    }
    assert aggregate["defined_only_macro"]["precision_attack"] == {
        "mean": 0.5,
        "n_defined": 1,
    }
    assert aggregate["pooled_confusion_matrix"] == {"tn": 12, "fp": 3, "fn": 4, "tp": 1}
    assert aggregate["pooled_metrics"]["precision_attack"] == 0.25


def test_targeted_holdout_resume_skips_independently_valid_completed_run(
    tmp_path: Path,
    synthetic_split: PreparedSplit,
    fake_model_factory,
):
    holdout = next(iter(EXPECTED_HOLDOUTS.values()))
    config = TargetedHoldoutWorkflowConfig(
        artifact_root=tmp_path / "runs",
        dataset_root=tmp_path / "dataset",
        cache_root=tmp_path / "cache",
        holdout_csvs=(holdout,),
        timesteps=2,
        resume=True,
    )
    calls: list[str] = []

    def execute(run_config):
        calls.append(run_config.run_id)
        return run_qrdqn_experiment(
            run_config,
            split_loader=lambda _config: synthetic_split,
            model_factory=fake_model_factory,
        )

    first = run_targeted_holdouts(config, executor=execute)
    run_id = next(iter(EXPECTED_HOLDOUTS))
    run_dir = config.artifact_root / run_id
    assert first["runs"][0]["execution"] == "completed"
    assert verify_artifact_manifest(run_dir)["status"] == "completed"

    second = run_targeted_holdouts(
        config,
        executor=lambda _config: pytest.fail("completed holdout must be skipped"),
    )
    assert second["runs"][0]["execution"] == "skipped_completed"
    assert calls == [run_id]
    summary = json.loads(config.summary_path.read_text(encoding="utf-8"))
    assert summary["holdout_csvs"] == [holdout]
