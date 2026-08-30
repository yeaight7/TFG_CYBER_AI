from __future__ import annotations

import inspect
from pathlib import Path

from scripts.preflight_gpu_environment import parse_args as parse_preflight_args
from scripts.run_campaign import parse_args as parse_campaign_args
from src.gpu_preflight import verify_preflight_report


REPO_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_ARTIFACT_ROOT = Path("runs/final_campaign")


def test_official_campaign_cli_defaults_use_repository_relative_artifact_root() -> None:
    campaign_args = parse_campaign_args(
        [
            "experiments/final_experiment_campaign.json",
            "--campaign-id",
            "campaign-test",
            "--cache-root",
            "cache/cicids2017",
            "--snapshot-root",
            "../campaign-exports",
            "--preflight-report",
            "runs/final_campaign/preflight_report.json",
            "--dry-run",
        ]
    )
    preflight_args = parse_preflight_args(
        [
            "--dataset-root",
            "datasets/CICIDS2017",
            "--cache-root",
            "cache/cicids2017",
            "--snapshot-root",
            "../campaign-exports",
        ]
    )

    assert campaign_args.artifact_root == OFFICIAL_ARTIFACT_ROOT
    assert preflight_args.artifact_root == OFFICIAL_ARTIFACT_ROOT
    assert not campaign_args.artifact_root.is_absolute()
    assert not preflight_args.artifact_root.is_absolute()


def test_maintained_campaign_examples_use_relative_artifacts_and_convenience_exports() -> None:
    guide = (REPO_ROOT / "docs" / "gpu_experimental_environment.md").read_text(
        encoding="utf-8"
    )
    reproducibility = (REPO_ROOT / "docs" / "reproducibility.md").read_text(
        encoding="utf-8"
    )

    assert "export ARTIFACT_ROOT=runs/final_campaign" in guide
    assert "ARTIFACT_ROOT=/" not in guide
    assert "different failure domain" not in guide.lower()
    assert "separate durable storage" not in reproducibility.lower()
    assert "manual download/recovery convenience" in reproducibility.lower()


def test_phase2_instructions_delegate_to_authoritative_git_lfs_policy() -> None:
    phase2 = (REPO_ROOT / "docs" / "phase 2" / "AGENT_CONTEXT.md").read_text(
        encoding="utf-8"
    )
    attributes = (REPO_ROOT / ".gitattributes").read_text(encoding="utf-8")

    assert "do not commit datasets or PCAPs" not in phase2
    assert ".gitattributes" in phase2
    assert ".gitignore" in phase2
    assert "Git LFS" in phase2
    assert "datasets/CICIDS2017/*.csv filter=lfs" in attributes
    assert "*.pcap filter=lfs" in attributes


def test_preflight_git_revision_remains_traceability_only() -> None:
    parameters = inspect.signature(verify_preflight_report).parameters
    verifier_source = inspect.getsource(verify_preflight_report)

    assert "expected_git_commit_sha" not in parameters
    assert "commit_sha" not in verifier_source
