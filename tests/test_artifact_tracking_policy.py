from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def _is_ignored(path: str) -> bool:
    return _git("check-ignore", "--quiet", "--no-index", path).returncode == 0


def _attribute(path: str, name: str) -> str:
    result = _git("check-attr", name, "--", path)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip().rsplit(": ", maxsplit=1)[-1]


def test_required_run_evidence_is_trackable_with_heavy_binaries_in_lfs() -> None:
    trackable = (
        "runs/final/attempt-1/model.zip",
        "runs/final/attempt-1/checkpoints/model_500000_steps.zip",
        "runs/final/attempt-1/tensorboard/events.out.tfevents.example",
        "runs/final/attempt-1/stdout.log",
        "runs/final/attempt-1/stderr.log",
        "runs/final/attempt-1/system_metrics.csv",
        "runs/final/attempt-1/feature_importances.csv",
        "runs/final/attempt-1/metrics.json",
    )
    assert all(not _is_ignored(path) for path in trackable)
    assert _attribute(trackable[0], "filter") == "lfs"
    assert _attribute(trackable[1], "filter") == "lfs"
    assert _attribute(trackable[2], "filter") == "lfs"
    assert _attribute(trackable[5], "filter") == "unspecified"
    assert _attribute(trackable[6], "filter") == "unspecified"


def test_future_prediction_csvs_and_transfer_bundles_are_ignored() -> None:
    assert _is_ignored("runs/final/attempt-1/predictions.csv")
    assert _is_ignored("runs/final/attempt-1/predictions_head_10000.csv")
    assert _is_ignored("exports/final-campaign.tar.gz")


def test_csv_lfs_scope_preserves_existing_large_evidence_only() -> None:
    assert (
        _attribute("datasets/CICIDS2017/Monday-WorkingHours.pcap_ISCX.csv", "filter")
        == "lfs"
    )
    assert _attribute("pcaps/archive/deprecated_lab_flows_benign.csv", "filter") == "lfs"
    assert _attribute("runs/phase2/example/predictions.csv", "filter") == "lfs"
    assert _attribute("runs/final/tensorboard_scalars/loss.csv", "filter") == "lfs"
    assert (
        _attribute("runs/final/plots/tensorboard_scalars/loss.csv", "filter") == "lfs"
    )
    assert (
        _attribute("runs/final/attempt-1/system_metrics.csv", "filter") == "unspecified"
    )


def test_existing_prediction_csvs_remain_in_the_index() -> None:
    existing = "runs/phase2/P2v2_pred_20260224_004121/predictions.csv"
    result = _git("ls-files", "--error-unmatch", existing)
    assert result.returncode == 0, result.stderr
