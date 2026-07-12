from __future__ import annotations

import copy
import importlib
import inspect
import json
from pathlib import Path
from subprocess import CompletedProcess, run
import sys

import pytest

from src.campaign import (
    CampaignArtifactError,
    CampaignDependencyError,
    CampaignPaths,
    CampaignRunner,
    CampaignSpecError,
    load_campaign_spec,
    validate_campaign_spec,
)
from src.cicids_cache import sha256_file
from src.run_artifacts import ArtifactManifestWriter, ArtifactRequirement, atomic_write_json


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = REPO_ROOT / "experiments" / "final_experiment_campaign.json"
FRESH_MAIN_ID = "qrdqn_main_random_full_s42_m42"


def _paths(tmp_path: Path, *, phase2_input: Path | None = None) -> CampaignPaths:
    return CampaignPaths(
        artifact_root=tmp_path / "artifacts",
        cache_root=tmp_path / "cache",
        dataset_root=tmp_path / "dataset",
        snapshot_root=tmp_path / "snapshots",
        preflight_report=tmp_path / "preflight.json",
        phase2_input=phase2_input,
        phase2_input_sha256=(
            None if phase2_input is None else sha256_file(phase2_input)
        ),
    )


def _valid_cache(_dataset_root: Path, cache_root: Path) -> dict[str, str]:
    cache_root.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_root / "cache_manifest.json"
    if not manifest_path.exists():
        atomic_write_json(manifest_path, {"validation_status": "valid"})
    return {
        "validation_status": "valid",
        "manifest_sha256": sha256_file(manifest_path),
    }


def _seal_attempt(
    entry,
    attempt_dir: Path,
    campaign_id: str,
    *,
    phase2_input: Path | None = None,
    source_manifest_override: str | None = None,
    config_override: dict | None = None,
) -> None:
    source_run_id = None
    source_manifest_sha256 = None
    if entry.logical_id != FRESH_MAIN_ID and entry.depends_on:
        main_dir = attempt_dir.parents[1] / FRESH_MAIN_ID / "attempt-1"
        source_run_id = "attempt-1"
        source_manifest_sha256 = (
            source_manifest_override
            or sha256_file(main_dir / "artifact_manifest.json")
        )
    writer = ArtifactManifestWriter(
        attempt_dir,
        run_metadata={
            "campaign_id": campaign_id,
            "logical_run_id": entry.logical_id,
            "physical_run_id": attempt_dir.name,
            "attempt": int(attempt_dir.name.rsplit("-", 1)[1]),
            "split_seed": entry.config.get("split_seed"),
            "model_seed": entry.config.get("model_seed"),
            "source_run_id": source_run_id,
            "source_manifest_sha256": source_manifest_sha256,
        },
        requirements={"config": ArtifactRequirement("config.json")},
    )
    writer.start()
    config = {
        "campaign_id": campaign_id,
        "logical_run_id": entry.logical_id,
        "run_id": attempt_dir.name,
        "profile_id": "main-v1",
        "split_seed": entry.config.get("split_seed"),
        "model_seed": entry.config.get("model_seed"),
        "timesteps": entry.config.get("timesteps"),
        "split_mode": entry.config.get("split_mode"),
        "train_max_rows": entry.config.get("train_max_rows"),
        "holdout_csv": entry.config.get("holdout_csv"),
        "source_run_id": source_run_id,
        "source_manifest_sha256": source_manifest_sha256,
    }
    if entry.logical_id == "phase2_fresh_main" and phase2_input is not None:
        config["input"] = {
            "filename": phase2_input.name,
            "size_bytes": phase2_input.stat().st_size,
            "sha256": sha256_file(phase2_input),
        }
    if config_override is not None:
        config.update(config_override)
    atomic_write_json(attempt_dir / "config.json", config)
    writer.complete()


def _successful_dispatch(
    calls: list[str],
    campaign_id: str,
    *,
    phase2_input: Path | None = None,
):
    def dispatch(entry, command, attempt_dir):
        calls.append(entry.logical_id)
        _seal_attempt(
            entry,
            attempt_dir,
            campaign_id,
            phase2_input=phase2_input,
        )
        return CompletedProcess(command, 0)

    return dispatch


def test_committed_spec_encodes_exact_locked_matrix() -> None:
    spec = load_campaign_spec(SPEC_PATH)

    assert spec.primary_logical_count == 24
    assert spec.primary_execution_count == 22
    assert spec.auxiliary_count == 5
    assert spec.alias_count == 2
    assert spec.aliases == {
        "qrdqn_ladder_full_s42_m42": FRESH_MAIN_ID,
        "qrdqn_seed_1m_s42_m42": "qrdqn_ladder_1m_s42_m42",
    }
    assert spec.stage_order == (
        "qrdqn_main",
        "main_direct_validation",
        "main_bootstrap_and_duplicate_analysis",
        "shuffled_label_validation",
        "phase2_fresh_main",
        "qrdqn_day",
        "qrdqn_ladder",
        "qrdqn_seed_sensitivity",
        "qrdqn_targeted_holdouts",
        "random_forest",
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload.update(schema_version="9.0"),
        lambda payload: payload["entries"].append(copy.deepcopy(payload["entries"][0])),
        lambda payload: payload["entries"].__delitem__(-1),
        lambda payload: payload["entries"][0].update(model_seed=999),
        lambda payload: payload["entries"][1].update(requires_fresh_main=False),
        lambda payload: payload["entries"][-1].update(stage="qrdqn_main"),
    ],
)
def test_schema_rejects_drift_from_locked_campaign(mutate) -> None:
    payload = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    mutate(payload)

    with pytest.raises(CampaignSpecError):
        validate_campaign_spec(payload)


def test_dry_run_is_side_effect_free_and_reports_exact_counts(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    runner = CampaignRunner(
        load_campaign_spec(SPEC_PATH),
        campaign_id="campaign-test",
        paths=paths,
        cache_validator=_valid_cache,
    )

    report = runner.dry_run()

    assert report["primary_model_training_executions"] == 22
    assert report["primary_training_logical_result_points"] == 24
    assert report["auxiliary_jobs"] == 5
    assert report["logical_aliases"] == 2
    assert report["dispatch_mode"] == "sequential"
    assert report["entries"][0]["logical_id"] == FRESH_MAIN_ID
    assert not paths.artifact_root.exists()
    assert not paths.cache_root.exists()


def test_selection_fails_without_dependencies_and_does_not_auto_dispatch(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    runner = CampaignRunner(
        load_campaign_spec(SPEC_PATH),
        campaign_id="campaign-test",
        paths=_paths(tmp_path),
        dispatcher=_successful_dispatch(calls, "campaign-test"),
        cache_validator=_valid_cache,
    )

    with pytest.raises(CampaignDependencyError, match="fresh campaign MAIN"):
        runner.execute(logical_run_id="main_direct_validation")

    assert calls == []


def test_full_dispatch_is_strictly_sequential_and_records_two_aliases(
    tmp_path: Path,
) -> None:
    flows = tmp_path / "lab-flows.csv"
    flows.write_text("Flow Duration,Label\n1,BENIGN\n", encoding="utf-8")
    calls: list[str] = []
    runner = CampaignRunner(
        load_campaign_spec(SPEC_PATH),
        campaign_id="campaign-test",
        paths=_paths(tmp_path, phase2_input=flows),
        dispatcher=_successful_dispatch(
            calls,
            "campaign-test",
            phase2_input=flows,
        ),
        cache_validator=_valid_cache,
    )

    report = runner.execute()

    assert len(calls) == 27
    assert len(set(calls)) == 27
    assert calls[0] == FRESH_MAIN_ID
    assert calls.index("qrdqn_day_full_s42_m42") < calls.index(
        "qrdqn_ladder_100k_s42_m42"
    )
    assert calls[-1] == "rf_holdout_ddos_m42"
    assert report["status"] == "completed"
    state = json.loads(runner.state_path.read_text(encoding="utf-8"))
    assert state["entries"]["qrdqn_ladder_full_s42_m42"]["status"] == "reused"
    assert state["entries"]["qrdqn_seed_1m_s42_m42"]["status"] == "reused"


def test_resume_skips_valid_completion_and_retries_failed_or_interrupted(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    campaign_id = "campaign-test"
    runner = CampaignRunner(
        load_campaign_spec(SPEC_PATH),
        campaign_id=campaign_id,
        paths=_paths(tmp_path),
        dispatcher=_successful_dispatch(calls, campaign_id),
        cache_validator=_valid_cache,
    )
    runner.execute(logical_run_id=FRESH_MAIN_ID)
    runner.execute(logical_run_id=FRESH_MAIN_ID, resume=True)
    assert calls == [FRESH_MAIN_ID]

    state = json.loads(runner.state_path.read_text(encoding="utf-8"))
    record = state["entries"][FRESH_MAIN_ID]
    record["status"] = "failed"
    atomic_write_json(runner.state_path, state)
    runner.execute(logical_run_id=FRESH_MAIN_ID, resume=True)
    assert calls == [FRESH_MAIN_ID, FRESH_MAIN_ID]
    assert runner.attempt_dir(FRESH_MAIN_ID, 2).is_dir()

    state = json.loads(runner.state_path.read_text(encoding="utf-8"))
    state["entries"][FRESH_MAIN_ID]["status"] = "running"
    atomic_write_json(runner.state_path, state)
    runner.execute(logical_run_id=FRESH_MAIN_ID, resume=True)
    assert calls == [FRESH_MAIN_ID, FRESH_MAIN_ID, FRESH_MAIN_ID]
    assert runner.attempt_dir(FRESH_MAIN_ID, 3).is_dir()


def test_invalid_completed_artifact_halts_before_next_dispatch(tmp_path: Path) -> None:
    calls: list[str] = []
    campaign_id = "campaign-test"

    def corrupt_main(entry, command, attempt_dir):
        calls.append(entry.logical_id)
        _seal_attempt(entry, attempt_dir, campaign_id)
        (attempt_dir / "config.json").write_text("tampered", encoding="utf-8")
        return CompletedProcess(command, 0)

    runner = CampaignRunner(
        load_campaign_spec(SPEC_PATH),
        campaign_id=campaign_id,
        paths=_paths(tmp_path),
        dispatcher=corrupt_main,
        cache_validator=_valid_cache,
    )

    with pytest.raises(CampaignArtifactError, match="invalid"):
        runner.execute(stage="qrdqn_main")

    assert calls == [FRESH_MAIN_ID]
    state = json.loads(runner.state_path.read_text(encoding="utf-8"))
    assert state["entries"][FRESH_MAIN_ID]["status"] == "invalid"


def test_primary_artifact_with_scientific_config_drift_is_invalid(tmp_path: Path) -> None:
    campaign_id = "campaign-test"

    def wrong_seed(entry, command, attempt_dir):
        _seal_attempt(
            entry,
            attempt_dir,
            campaign_id,
            config_override={"model_seed": 999},
        )
        return CompletedProcess(command, 0)

    runner = CampaignRunner(
        load_campaign_spec(SPEC_PATH),
        campaign_id=campaign_id,
        paths=_paths(tmp_path),
        dispatcher=wrong_seed,
        cache_validator=_valid_cache,
    )

    with pytest.raises(CampaignArtifactError, match="scientific config"):
        runner.execute(logical_run_id=FRESH_MAIN_ID)


def test_fresh_main_gate_rejects_historical_or_foreign_manifest(tmp_path: Path) -> None:
    campaign_id = "campaign-test"
    calls: list[str] = []
    runner = CampaignRunner(
        load_campaign_spec(SPEC_PATH),
        campaign_id=campaign_id,
        paths=_paths(tmp_path),
        dispatcher=_successful_dispatch(calls, campaign_id),
        cache_validator=_valid_cache,
    )
    runner.execute(logical_run_id=FRESH_MAIN_ID)
    manifest_path = runner.attempt_dir(FRESH_MAIN_ID, 1) / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["run"]["campaign_id"] = None
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(CampaignDependencyError, match="fresh campaign MAIN"):
        runner.execute(logical_run_id="main_direct_validation", resume=True)

    assert calls == [FRESH_MAIN_ID]


def test_phase2_requires_matching_validated_input_hash(tmp_path: Path) -> None:
    campaign_id = "campaign-test"
    calls: list[str] = []
    runner = CampaignRunner(
        load_campaign_spec(SPEC_PATH),
        campaign_id=campaign_id,
        paths=_paths(tmp_path),
        dispatcher=_successful_dispatch(calls, campaign_id),
        cache_validator=_valid_cache,
    )
    runner.execute(logical_run_id=FRESH_MAIN_ID)

    with pytest.raises(CampaignDependencyError, match="Phase 2 input hash"):
        runner.execute(logical_run_id="phase2_fresh_main", resume=True)


def test_auxiliary_artifact_must_bind_to_fresh_main_manifest(tmp_path: Path) -> None:
    campaign_id = "campaign-test"
    calls: list[str] = []
    paths = _paths(tmp_path)
    runner = CampaignRunner(
        load_campaign_spec(SPEC_PATH),
        campaign_id=campaign_id,
        paths=paths,
        dispatcher=_successful_dispatch(calls, campaign_id),
        cache_validator=_valid_cache,
    )
    runner.execute(logical_run_id=FRESH_MAIN_ID)

    def foreign_source(entry, command, attempt_dir):
        calls.append(entry.logical_id)
        _seal_attempt(
            entry,
            attempt_dir,
            campaign_id,
            source_manifest_override="0" * 64,
        )
        return CompletedProcess(command, 0)

    resumed = CampaignRunner(
        load_campaign_spec(SPEC_PATH),
        campaign_id=campaign_id,
        paths=paths,
        dispatcher=foreign_source,
        cache_validator=_valid_cache,
    )
    with pytest.raises(CampaignArtifactError, match="fresh campaign MAIN"):
        resumed.execute(logical_run_id="main_direct_validation", resume=True)


def test_campaign_cli_dry_run_reports_contract_without_writing_artifacts(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "artifacts"
    result = run(
        [
            "uv",
            "run",
            "python",
            "scripts/run_campaign.py",
            str(SPEC_PATH),
            "--campaign-id",
            "campaign-test",
            "--artifact-root",
            str(artifact_root),
            "--cache-root",
            str(tmp_path / "cache"),
            "--snapshot-root",
            str(tmp_path / "snapshots"),
            "--preflight-report",
            str(tmp_path / "preflight.json"),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["primary_model_training_executions"] == 22
    assert report["auxiliary_jobs"] == 5
    assert report["logical_aliases"] == 2
    assert not artifact_root.exists()


@pytest.mark.parametrize(
    ("module_name", "required_args"),
    [
        ("src.train_rl_defender", ["--profile", "main-v1"]),
        (
            "src.baseline_random_forest",
            [
                "--split-mode",
                "random",
                "--cache-root",
                "cache",
                "--artifact-root",
                "artifacts",
                "--run-id",
                "attempt-2",
            ],
        ),
        ("src.validate_checks", []),
        (
            "scripts.validate_main_direct",
            [
                "--run-dir",
                "main",
                "--artifact-root",
                "artifacts",
                "--job-id",
                "attempt-2",
            ],
        ),
        (
            "scripts.bootstrap_ci",
            ["--run-dir", "main", "--output-dir", "attempt-2"],
        ),
        (
            "scripts.analyze_duplicates",
            ["--run-dir", "main", "--output-dir", "attempt-2"],
        ),
        ("scripts.predict_real_traffic_v2", ["--flows", "flows.csv"]),
    ],
)
def test_dispatched_entrypoints_accept_campaign_identity(
    module_name: str,
    required_args: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if str(REPO_ROOT / "src") not in sys.path:
        monkeypatch.syspath_prepend(str(REPO_ROOT / "src"))
    module = importlib.import_module(module_name)
    parse_args = getattr(module, "parse_args")
    argv = [
        *required_args,
        "--campaign-id",
        "campaign-test",
        "--logical-run-id",
        "logical-test",
        "--attempt",
        "2",
    ]
    if len(inspect.signature(parse_args).parameters) == 0:
        monkeypatch.setattr("sys.argv", [module_name, *argv])
        args = parse_args()
    else:
        args = parse_args(argv)

    assert args.campaign_id == "campaign-test"
    assert args.logical_run_id == "logical-test"
    assert args.attempt == 2


def test_shuffled_campaign_dispatch_avoids_legacy_validation_wrapper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("src.validate_checks")
    observed = []

    def fake_run(config):
        observed.append(config)
        return config.artifact_root / config.run_id

    legacy_root = tmp_path / "legacy-runs"
    monkeypatch.setattr(module, "RUNS_DIR", legacy_root)
    monkeypatch.setattr(module, "run_shuffled_label_validation", fake_run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "validate_checks.py",
            "--checks",
            "B",
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--cache-root",
            str(tmp_path / "cache"),
            "--artifact-root",
            str(tmp_path / "campaign-attempts"),
            "--run-id-b",
            "attempt-1",
            "--campaign-id",
            "campaign-test",
            "--logical-run-id",
            "shuffled_label_validation_s42_m42",
            "--attempt",
            "1",
        ],
    )

    module.main()

    assert len(observed) == 1
    assert observed[0].campaign_id == "campaign-test"
    assert observed[0].logical_run_id == "shuffled_label_validation_s42_m42"
    assert not legacy_root.exists()


def test_qrdqn_cli_forwards_campaign_identity_to_single_run_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if str(REPO_ROOT / "src") not in sys.path:
        monkeypatch.syspath_prepend(str(REPO_ROOT / "src"))
    module = importlib.import_module("src.train_rl_defender")
    qrdqn_module = importlib.import_module("src.qrdqn_experiment")
    observed = []
    monkeypatch.setattr(
        qrdqn_module,
        "run_qrdqn_experiment",
        lambda config: observed.append(config),
    )
    args = module.parse_args(
        [
            "--profile",
            "main-v1",
            "--cache-root",
            str(tmp_path / "cache"),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--run-id",
            "attempt-2",
            "--campaign-id",
            "campaign-test",
            "--logical-run-id",
            FRESH_MAIN_ID,
            "--attempt",
            "2",
        ]
    )

    module._phase4_main(args)

    assert len(observed) == 1
    assert observed[0].campaign_id == "campaign-test"
    assert observed[0].logical_run_id == FRESH_MAIN_ID
    assert observed[0].attempt == 2


def test_spec_and_commands_use_provider_neutral_paths(tmp_path: Path) -> None:
    spec_text = SPEC_PATH.read_text(encoding="utf-8").lower()
    assert "runpod" not in spec_text
    assert "vast.ai" not in spec_text
    assert "lambda labs" not in spec_text
    assert "c:\\" not in spec_text
    runner = CampaignRunner(
        load_campaign_spec(SPEC_PATH),
        campaign_id="campaign-test",
        paths=_paths(tmp_path),
        cache_validator=_valid_cache,
    )
    command = runner.command_for(FRESH_MAIN_ID, attempt=1)
    assert str(tmp_path) in " ".join(command)
