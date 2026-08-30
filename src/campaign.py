"""Locked sequential campaign specification, state, and subprocess orchestration."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from src.artifact_integrity import ArtifactTrustError, verify_artifact_manifest
from src.campaign_export import create_final_bundle, create_run_export
from src.cicids_cache import sha256_file, validate_cache
from src.gpu_preflight import verify_preflight_report
from src.run_artifacts import atomic_write_json


_REPO_ROOT = Path(__file__).resolve().parent.parent
CAMPAIGN_STATE_SCHEMA_VERSION = "1.0"
CAMPAIGN_SPEC_SCHEMA_VERSION = "1.0"
DEFAULT_CAMPAIGN_ARTIFACT_ROOT = Path("runs/final_campaign")
FRESH_MAIN_ID = "qrdqn_main_random_full_s42_m42"
STAGE_ORDER = (
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
APPROVED_ALIASES = {
    "qrdqn_ladder_full_s42_m42": FRESH_MAIN_ID,
    "qrdqn_seed_1m_s42_m42": "qrdqn_ladder_1m_s42_m42",
}
_AUXILIARY_CONTRACTS = {
    "main_direct_validation": (
        "main_direct_validation",
        "main_direct_validation",
        (FRESH_MAIN_ID,),
    ),
    "main_bootstrap_ci": (
        "main_bootstrap_and_duplicate_analysis",
        "bootstrap_ci",
        (FRESH_MAIN_ID, "main_direct_validation"),
    ),
    "main_duplicate_analysis": (
        "main_bootstrap_and_duplicate_analysis",
        "duplicate_analysis",
        (FRESH_MAIN_ID,),
    ),
    "shuffled_label_validation_s42_m42": (
        "shuffled_label_validation",
        "shuffled_label_validation",
        (),
    ),
    "phase2_fresh_main": (
        "phase2_fresh_main",
        "phase2_inference",
        (FRESH_MAIN_ID,),
    ),
}
_TARGETED_HOLDOUTS = {
    "webattacks": "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "infilteration": "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "portscan": "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "ddos": "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
}


class CampaignError(RuntimeError):
    """Base campaign failure."""


class CampaignSpecError(CampaignError):
    """Campaign specification is malformed or drifts from the approved matrix."""


class CampaignDependencyError(CampaignError):
    """A selected entry lacks validated prerequisite evidence."""


class CampaignArtifactError(CampaignError):
    """A subprocess did not produce compatible validated evidence."""


class CampaignExecutionError(CampaignError):
    """A sequential subprocess returned a non-zero status."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _safe_component(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise CampaignSpecError(f"{field} must be one non-empty path component")
    return value


def _expected_primary_contracts() -> dict[str, tuple[Any, ...]]:
    contracts: dict[str, tuple[Any, ...]] = {}

    def add(
        logical_id: str,
        stage: str,
        runner: str,
        split_mode: str,
        split_seed: int,
        model_seed: int,
        timesteps: int | None,
        train_max_rows: int | None,
        holdout_csv: str | None,
    ) -> None:
        contracts[logical_id] = (
            stage,
            runner,
            split_mode,
            split_seed,
            model_seed,
            timesteps,
            train_max_rows,
            holdout_csv,
        )

    add(FRESH_MAIN_ID, "qrdqn_main", "qrdqn", "random", 42, 42, 3_000_000, None, None)
    add(
        "qrdqn_day_full_s42_m42",
        "qrdqn_day",
        "qrdqn",
        "day",
        42,
        42,
        3_000_000,
        None,
        None,
    )
    for rows, timesteps, label in (
        (100_000, 132_474, "100k"),
        (250_000, 331_185, "250k"),
        (500_000, 662_370, "500k"),
        (1_000_000, 1_324_741, "1m"),
        (2_000_000, 2_649_482, "2m"),
    ):
        add(
            f"qrdqn_ladder_{label}_s42_m42",
            "qrdqn_ladder",
            "qrdqn",
            "random",
            42,
            42,
            timesteps,
            rows,
            None,
        )
    for seed in range(43, 47):
        add(
            f"qrdqn_seed_1m_s42_m{seed}",
            "qrdqn_seed_sensitivity",
            "qrdqn",
            "random",
            42,
            seed,
            1_324_741,
            1_000_000,
            None,
        )
    for label, csv_name in _TARGETED_HOLDOUTS.items():
        add(
            f"qrdqn_holdout_{label}_m42",
            "qrdqn_targeted_holdouts",
            "qrdqn",
            "exact-holdout",
            42,
            42,
            1_000_000,
            None,
            csv_name,
        )
    add(
        "rf_random_full_s42_m42",
        "random_forest",
        "random_forest",
        "random",
        42,
        42,
        None,
        None,
        None,
    )
    add(
        "rf_random_1m_s42_m42",
        "random_forest",
        "random_forest",
        "random",
        42,
        42,
        None,
        1_000_000,
        None,
    )
    add(
        "rf_day_full_s42_m42",
        "random_forest",
        "random_forest",
        "day",
        42,
        42,
        None,
        None,
        None,
    )
    for label, csv_name in _TARGETED_HOLDOUTS.items():
        add(
            f"rf_holdout_{label}_m42",
            "random_forest",
            "random_forest",
            "exact-holdout",
            42,
            42,
            None,
            None,
            csv_name,
        )
    return contracts


_PRIMARY_CONTRACTS = _expected_primary_contracts()


@dataclass(frozen=True)
class CampaignEntry:
    logical_id: str
    classification: str
    stage: str
    runner: str | None
    config: Mapping[str, Any]

    @property
    def depends_on(self) -> tuple[str, ...]:
        value = self.config.get("depends_on", ())
        return tuple(str(item) for item in value)

    @property
    def reuse_of(self) -> str | None:
        value = self.config.get("reuse_of")
        return None if value is None else str(value)


@dataclass(frozen=True)
class CampaignSpec:
    raw: Mapping[str, Any]
    entries: tuple[CampaignEntry, ...]
    stage_order: tuple[str, ...]
    content_hash: str

    @property
    def by_id(self) -> dict[str, CampaignEntry]:
        return {entry.logical_id: entry for entry in self.entries}

    @property
    def aliases(self) -> dict[str, str]:
        return {
            entry.logical_id: str(entry.reuse_of)
            for entry in self.entries
            if entry.classification == "alias"
        }

    @property
    def alias_count(self) -> int:
        return len(self.aliases)

    @property
    def auxiliary_count(self) -> int:
        return sum(entry.classification == "auxiliary" for entry in self.entries)

    @property
    def primary_execution_count(self) -> int:
        return sum(
            entry.classification == "primary_model_training" for entry in self.entries
        )

    @property
    def primary_logical_count(self) -> int:
        return self.primary_execution_count + self.alias_count


@dataclass(frozen=True)
class CampaignPaths:
    artifact_root: Path
    cache_root: Path
    dataset_root: Path
    snapshot_root: Path | None = None
    preflight_report: Path | None = None
    phase2_input: Path | None = None
    phase2_input_sha256: str | None = None
    repository_root: Path = _REPO_ROOT

    def __post_init__(self) -> None:
        repository_root = Path(self.repository_root).resolve()
        object.__setattr__(self, "repository_root", repository_root)
        for field in (
            "cache_root",
            "dataset_root",
            "snapshot_root",
            "preflight_report",
            "phase2_input",
        ):
            value = getattr(self, field)
            if value is not None:
                path = Path(value)
                if not path.is_absolute():
                    path = repository_root / path
                object.__setattr__(self, field, path.resolve())
        artifact_root = Path(self.artifact_root)
        if not artifact_root.is_absolute():
            artifact_root = repository_root / artifact_root
        artifact_root = artifact_root.resolve()
        try:
            artifact_root.relative_to(repository_root)
        except ValueError as error:
            raise ValueError(
                "Official campaign artifact_root must live beneath the repository"
            ) from error
        object.__setattr__(self, "artifact_root", artifact_root)
        if self.snapshot_root is not None:
            try:
                self.snapshot_root.relative_to(repository_root)
            except ValueError:
                pass
            else:
                raise ValueError("snapshot_root must remain outside the repository")
        if self.phase2_input_sha256 is not None:
            digest = self.phase2_input_sha256.lower()
            if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                raise ValueError("phase2_input_sha256 must be a SHA-256 hex digest")
            object.__setattr__(self, "phase2_input_sha256", digest)


def validate_campaign_spec(payload: Mapping[str, Any]) -> CampaignSpec:
    if not isinstance(payload, Mapping):
        raise CampaignSpecError("Campaign specification must be a JSON object")
    if payload.get("schema_version") != CAMPAIGN_SPEC_SCHEMA_VERSION:
        raise CampaignSpecError("Unsupported campaign specification schema_version")
    if payload.get("campaign_spec_id") != "final-experiment-v1":
        raise CampaignSpecError("campaign_spec_id must be 'final-experiment-v1'")
    if payload.get("profile_id") != "main-v1":
        raise CampaignSpecError("profile_id must be 'main-v1'")
    expected_profile_hash = (
        "17bbeb3f8020f7a1f8860e70b9fbf65b495f71d3dc40e1e30f24dfa86299a19a"
    )
    if payload.get("profile_hash") != expected_profile_hash:
        raise CampaignSpecError("main-v1 profile_hash does not match the frozen profile")
    if payload.get("cache_policy") != "require":
        raise CampaignSpecError("Official campaign cache_policy must be 'require'")
    dataset_root = payload.get("dataset_root")
    if not isinstance(dataset_root, str) or Path(dataset_root).is_absolute():
        raise CampaignSpecError("dataset_root must be a provider-neutral relative path")
    if tuple(payload.get("stage_order", ())) != STAGE_ORDER:
        raise CampaignSpecError("Campaign stage_order differs from the approved sequence")
    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list):
        raise CampaignSpecError("entries must be a JSON array")

    entries: list[CampaignEntry] = []
    seen: set[str] = set()
    previous_stage = -1
    stage_positions = {stage: index for index, stage in enumerate(STAGE_ORDER)}
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, Mapping):
            raise CampaignSpecError("Each campaign entry must be a JSON object")
        logical_id = _safe_component(raw_entry.get("logical_id"), "logical_id")
        if logical_id in seen:
            raise CampaignSpecError(f"Duplicate logical_id: {logical_id}")
        seen.add(logical_id)
        classification = raw_entry.get("classification")
        if classification not in {"primary_model_training", "auxiliary", "alias"}:
            raise CampaignSpecError(f"Invalid classification for {logical_id}")
        stage = raw_entry.get("stage")
        if stage not in stage_positions:
            raise CampaignSpecError(f"Invalid stage for {logical_id}: {stage!r}")
        if stage_positions[stage] < previous_stage:
            raise CampaignSpecError("Campaign entries must follow stage_order")
        previous_stage = stage_positions[stage]
        runner = raw_entry.get("runner")
        if runner is not None and not isinstance(runner, str):
            raise CampaignSpecError(f"Invalid runner for {logical_id}")
        entries.append(
            CampaignEntry(
                logical_id=logical_id,
                classification=str(classification),
                stage=str(stage),
                runner=runner,
                config=dict(raw_entry),
            )
        )

    by_id = {entry.logical_id: entry for entry in entries}
    if set(by_id).difference(
        set(_PRIMARY_CONTRACTS) | set(APPROVED_ALIASES) | set(_AUXILIARY_CONTRACTS)
    ):
        raise CampaignSpecError("Campaign contains an unapproved logical entry")
    if set(_PRIMARY_CONTRACTS).difference(by_id):
        raise CampaignSpecError("Campaign is missing approved primary executions")
    if set(APPROVED_ALIASES).difference(by_id):
        raise CampaignSpecError("Campaign is missing approved aliases")
    if set(_AUXILIARY_CONTRACTS).difference(by_id):
        raise CampaignSpecError("Campaign is missing approved auxiliary jobs")

    for logical_id, expected in _PRIMARY_CONTRACTS.items():
        entry = by_id[logical_id]
        actual = (
            entry.stage,
            entry.runner,
            entry.config.get("split_mode"),
            entry.config.get("split_seed"),
            entry.config.get("model_seed"),
            entry.config.get("timesteps"),
            entry.config.get("train_max_rows"),
            entry.config.get("holdout_csv"),
        )
        if entry.classification != "primary_model_training" or actual != expected:
            raise CampaignSpecError(f"Primary execution contract drift: {logical_id}")
        if entry.config.get("requires_cache") is not True:
            raise CampaignSpecError(f"Primary execution must require cache: {logical_id}")

    for logical_id, reuse_of in APPROVED_ALIASES.items():
        entry = by_id[logical_id]
        if (
            entry.classification != "alias"
            or entry.reuse_of != reuse_of
            or entry.config.get("logical_result_classification")
            != "primary_model_training"
        ):
            raise CampaignSpecError(f"Alias contract drift: {logical_id}")

    for logical_id, (stage, runner, dependencies) in _AUXILIARY_CONTRACTS.items():
        entry = by_id[logical_id]
        if (
            entry.classification != "auxiliary"
            or entry.stage != stage
            or entry.runner != runner
            or entry.depends_on != dependencies
        ):
            raise CampaignSpecError(f"Auxiliary job contract drift: {logical_id}")
        if logical_id != "shuffled_label_validation_s42_m42":
            if entry.config.get("requires_fresh_main") is not True:
                raise CampaignSpecError(
                    f"Auxiliary job must require the fresh campaign MAIN: {logical_id}"
                )
    if by_id["main_duplicate_analysis"].config.get("requires_cache") is not True:
        raise CampaignSpecError("Duplicate analysis must require the validated cache")
    shuffled = by_id["shuffled_label_validation_s42_m42"].config
    if (
        shuffled.get("timesteps") != 10_000
        or shuffled.get("split_seed") != 42
        or shuffled.get("model_seed") != 42
        or shuffled.get("shuffled_label_seed") != 42
        or shuffled.get("requires_cache") is not True
        or shuffled.get("requires_fixed_split") is not True
    ):
        raise CampaignSpecError("Shuffled-label auxiliary contract drift")
    bootstrap = by_id["main_bootstrap_ci"].config
    if bootstrap.get("n_resamples") != 10_000 or bootstrap.get("bootstrap_seed") != 12_345:
        raise CampaignSpecError("Bootstrap auxiliary contract drift")
    if by_id["phase2_fresh_main"].config.get("requires_phase2_input") is not True:
        raise CampaignSpecError("Phase 2 auxiliary must require a validated input")

    spec = CampaignSpec(
        raw=dict(payload),
        entries=tuple(entries),
        stage_order=STAGE_ORDER,
        content_hash=hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest(),
    )
    if (
        spec.primary_execution_count != 22
        or spec.primary_logical_count != 24
        or spec.auxiliary_count != 5
        or spec.alias_count != 2
    ):
        raise CampaignSpecError("Campaign counts differ from the approved 22/24/5/2 contract")
    return spec


def load_campaign_spec(path: Path | str) -> CampaignSpec:
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CampaignSpecError(f"Cannot read campaign specification: {path}") from error
    return validate_campaign_spec(payload)


Dispatch = Callable[[CampaignEntry, Sequence[str], Path], subprocess.CompletedProcess[Any]]
CacheValidator = Callable[[Path, Path], Mapping[str, Any]]
PreflightValidator = Callable[..., Mapping[str, Any]]
RunExporter = Callable[..., Mapping[str, Any]]
BundleExporter = Callable[[Path, Path], Mapping[str, Any]]


def _copy_file_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _dispatch_subprocess(
    _entry: CampaignEntry,
    command: Sequence[str],
    _attempt_dir: Path,
) -> subprocess.CompletedProcess[Any]:
    return subprocess.run(list(command), check=False, cwd=_REPO_ROOT)


class CampaignRunner:
    """Execute one approved campaign entry at a time with atomic run-level state."""

    def __init__(
        self,
        spec: CampaignSpec,
        *,
        campaign_id: str,
        paths: CampaignPaths,
        dispatcher: Dispatch = _dispatch_subprocess,
        cache_validator: CacheValidator = validate_cache,
        preflight_validator: PreflightValidator = verify_preflight_report,
        run_exporter: RunExporter = create_run_export,
        bundle_exporter: BundleExporter = create_final_bundle,
    ) -> None:
        self.spec = spec
        self.campaign_id = _safe_component(campaign_id, "campaign_id")
        self.paths = paths
        self.dispatcher = dispatcher
        self.cache_validator = cache_validator
        self.preflight_validator = preflight_validator
        self.run_exporter = run_exporter
        self.bundle_exporter = bundle_exporter
        self.campaign_dir = self.paths.artifact_root / self.campaign_id
        self.state_path = self.campaign_dir / "campaign_state.json"
        self._preflight: dict[str, Any] | None = None
        self._preflight_report_sha256: str | None = None
        self._validated_cache_sha256: str | None = None

    def attempt_dir(self, logical_id: str, attempt: int) -> Path:
        return self.campaign_dir / "attempts" / logical_id / f"attempt-{attempt}"

    def _repository_relative(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.paths.repository_root).as_posix()
        except ValueError as error:
            raise CampaignExecutionError(
                f"Campaign artifact path is outside the repository: {path}"
            ) from error

    def _selected(
        self,
        *,
        stage: str | None = None,
        logical_run_id: str | None = None,
    ) -> tuple[CampaignEntry, ...]:
        if stage is not None and logical_run_id is not None:
            raise CampaignSpecError("Select either --stage or --run, not both")
        if stage is not None:
            if stage not in self.spec.stage_order:
                raise CampaignSpecError(f"Unknown campaign stage: {stage}")
            return tuple(entry for entry in self.spec.entries if entry.stage == stage)
        if logical_run_id is not None:
            entry = self.spec.by_id.get(logical_run_id)
            if entry is None:
                raise CampaignSpecError(f"Unknown logical run/job: {logical_run_id}")
            return (entry,)
        return self.spec.entries

    def dry_run(
        self,
        *,
        stage: str | None = None,
        logical_run_id: str | None = None,
    ) -> dict[str, Any]:
        selected = self._selected(stage=stage, logical_run_id=logical_run_id)
        entries = []
        for entry in selected:
            item = {
                "logical_id": entry.logical_id,
                "stage": entry.stage,
                "classification": entry.classification,
                "runner": entry.runner,
            }
            if entry.classification == "alias":
                item["reuse_of"] = entry.reuse_of
            else:
                item["command"] = self.command_for(entry.logical_id, attempt=1)
            entries.append(item)
        return {
            "mode": "dry-run",
            "campaign_id": self.campaign_id,
            "campaign_spec_sha256": self.spec.content_hash,
            "dispatch_mode": "sequential",
            "selection": {"stage": stage, "logical_run_id": logical_run_id},
            "primary_model_training_executions": self.spec.primary_execution_count,
            "primary_training_logical_result_points": self.spec.primary_logical_count,
            "auxiliary_jobs": self.spec.auxiliary_count,
            "logical_aliases": self.spec.alias_count,
            "entries": entries,
        }

    def _validate_preflight_gate(self) -> None:
        if self.paths.preflight_report is None:
            raise CampaignDependencyError("A successful matching preflight report is required")
        if not self.paths.preflight_report.is_file():
            raise CampaignDependencyError(
                f"A successful matching preflight report is required: "
                f"missing {self.paths.preflight_report}"
            )
        if self.paths.snapshot_root is None:
            raise CampaignDependencyError("A snapshot root is required for real campaign execution")
        try:
            cache_result = dict(
                self.cache_validator(self.paths.dataset_root, self.paths.cache_root)
            )
        except Exception as error:
            raise CampaignDependencyError(
                f"A successful matching preflight requires a validated cache: {error}"
            ) from error
        if cache_result.get("validation_status") != "valid":
            raise CampaignDependencyError(
                "A successful matching preflight requires a validated cache"
            )
        cache_manifest = self.paths.cache_root / "cache_manifest.json"
        cache_sha256 = cache_result.get("manifest_sha256")
        if cache_sha256 is None:
            if not cache_manifest.is_file():
                raise CampaignDependencyError("Validated cache manifest is missing")
            cache_sha256 = sha256_file(cache_manifest)
        try:
            report = dict(
                self.preflight_validator(
                    self.paths.preflight_report,
                    expected_campaign_spec_sha256=self.spec.content_hash,
                    expected_dataset_root=self.paths.dataset_root,
                    expected_cache_root=self.paths.cache_root,
                    expected_artifact_root=self.paths.artifact_root,
                    expected_snapshot_root=self.paths.snapshot_root,
                    expected_cache_manifest_sha256=str(cache_sha256),
                    expected_phase2_input_sha256=self.paths.phase2_input_sha256,
                    repository_root=self.paths.repository_root,
                )
            )
        except Exception as error:
            raise CampaignDependencyError(
                f"A successful matching preflight report is required: {error}"
            ) from error
        if report.get("status") != "passed":
            raise CampaignDependencyError("A successful matching preflight report is required")
        report_sha256 = report.get("report_sha256")
        if not isinstance(report_sha256, str) or len(report_sha256) != 64:
            raise CampaignDependencyError("Preflight report has no valid content hash")
        self._preflight = report
        self._preflight_report_sha256 = report_sha256
        self._validated_cache_sha256 = str(cache_sha256)

    def _write_campaign_inputs(self) -> None:
        if (
            self._preflight is None
            or self._preflight_report_sha256 is None
            or self._validated_cache_sha256 is None
        ):
            raise CampaignExecutionError("Preflight gate was not evaluated")
        cache_manifest = self.paths.cache_root / "cache_manifest.json"
        if not cache_manifest.is_file():
            raise CampaignExecutionError("Validated cache manifest disappeared")
        resolved_spec = {
            **dict(self.spec.raw),
            "resolved_runtime": {
                "artifact_root": self._repository_relative(self.paths.artifact_root),
                "cache_root": str(self.paths.cache_root.resolve()),
                "dataset_root": str(self.paths.dataset_root.resolve()),
                "snapshot_root": str(self.paths.snapshot_root.resolve()),
                "phase2_input": (
                    None
                    if self.paths.phase2_input is None
                    else str(self.paths.phase2_input.resolve())
                ),
                "phase2_input_sha256": self.paths.phase2_input_sha256,
                "preflight_report_sha256": self._preflight_report_sha256,
                "cache_manifest_sha256": self._validated_cache_sha256,
            },
        }
        atomic_write_json(self.campaign_dir / "campaign_spec_original.json", self.spec.raw)
        atomic_write_json(self.campaign_dir / "campaign_spec_resolved.json", resolved_spec)
        atomic_write_json(self.campaign_dir / "preflight_report.json", self._preflight)
        _copy_file_atomic(cache_manifest, self.campaign_dir / "cache_manifest.json")

    def _state_template(self) -> dict[str, Any]:
        return {
            "schema_version": CAMPAIGN_STATE_SCHEMA_VERSION,
            "campaign_id": self.campaign_id,
            "campaign_spec_id": self.spec.raw["campaign_spec_id"],
            "campaign_spec_sha256": self.spec.content_hash,
            "dispatch_mode": "sequential",
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "cache_manifest_sha256": self._validated_cache_sha256,
            "preflight_report_sha256": self._preflight_report_sha256,
            "entries": {
                entry.logical_id: {
                    "classification": entry.classification,
                    "stage": entry.stage,
                    "status": "pending",
                    "attempts": [],
                    **(
                        {"reuse_of": entry.reuse_of}
                        if entry.classification == "alias"
                        else {}
                    ),
                }
                for entry in self.spec.entries
            },
        }

    def _write_state(self, state: dict[str, Any]) -> None:
        state["updated_at"] = _utc_now()
        atomic_write_json(self.state_path, state)

    def _load_state(self, *, resume: bool) -> dict[str, Any]:
        if self.state_path.is_file():
            if not resume:
                raise CampaignExecutionError(
                    "Campaign state already exists; use --resume to preserve completed evidence"
                )
            try:
                state = json.loads(self.state_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:
                raise CampaignExecutionError("Campaign state JSON is invalid") from error
            if (
                state.get("schema_version") != CAMPAIGN_STATE_SCHEMA_VERSION
                or state.get("campaign_id") != self.campaign_id
                or state.get("campaign_spec_sha256") != self.spec.content_hash
            ):
                raise CampaignExecutionError("Campaign state identity does not match this run")
            if (
                state.get("preflight_report_sha256") != self._preflight_report_sha256
                or state.get("cache_manifest_sha256") != self._validated_cache_sha256
            ):
                raise CampaignExecutionError(
                    "Campaign state preflight/cache binding does not match this run"
                )
            for relative in (
                "campaign_spec_original.json",
                "campaign_spec_resolved.json",
                "preflight_report.json",
                "cache_manifest.json",
            ):
                if not (self.campaign_dir / relative).is_file():
                    raise CampaignExecutionError(
                        f"Campaign evidence input is missing: {relative}"
                    )
            return state
        self.campaign_dir.mkdir(parents=True, exist_ok=False)
        self._write_campaign_inputs()
        state = self._state_template()
        self._write_state(state)
        return state

    def _relative_attempt(self, path: Path) -> str:
        return path.relative_to(self.campaign_dir).as_posix()

    def _record_attempt_path(self, record: Mapping[str, Any]) -> Path:
        attempts = record.get("attempts", [])
        if not attempts:
            raise CampaignDependencyError("Completed campaign entry has no attempt record")
        relative = Path(str(attempts[-1]["artifact_dir"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise CampaignDependencyError("Campaign state artifact path is not provider-neutral")
        resolved = (self.campaign_dir / relative).resolve()
        try:
            resolved.relative_to(self.campaign_dir.resolve())
        except ValueError as error:
            raise CampaignDependencyError("Campaign state artifact path escapes campaign") from error
        return resolved

    def _verify_identity(
        self,
        entry: CampaignEntry,
        attempt_dir: Path,
        *,
        attempt: int,
    ) -> dict[str, Any]:
        try:
            verify_artifact_manifest(attempt_dir)
            manifest = json.loads(
                (attempt_dir / "artifact_manifest.json").read_text(encoding="utf-8")
            )
        except (ArtifactTrustError, OSError, json.JSONDecodeError) as error:
            raise CampaignArtifactError(
                f"Campaign artifact is invalid for {entry.logical_id}: {error}"
            ) from error
        run = manifest.get("run")
        if not isinstance(run, Mapping):
            raise CampaignArtifactError(f"Campaign artifact is invalid for {entry.logical_id}")
        expected = {
            "campaign_id": self.campaign_id,
            "logical_run_id": entry.logical_id,
            "physical_run_id": attempt_dir.name,
            "attempt": attempt,
        }
        mismatched = {key: value for key, value in expected.items() if run.get(key) != value}
        if mismatched:
            raise CampaignArtifactError(
                f"Campaign artifact is invalid for {entry.logical_id}: identity {mismatched}"
            )
        if entry.classification == "primary_model_training":
            config = json.loads(
                (attempt_dir / "config.json").read_text(encoding="utf-8")
            )
            request = config.get("request")
            if not isinstance(request, Mapping):
                request = config

            def resolved_value(key: str) -> Any:
                return config[key] if key in config else request.get(key)

            expected_config = {
                "split_mode": entry.config.get("split_mode"),
                "split_seed": entry.config.get("split_seed"),
                "model_seed": entry.config.get("model_seed"),
                "train_max_rows": entry.config.get("train_max_rows"),
                "holdout_csv": entry.config.get("holdout_csv"),
            }
            if entry.runner == "qrdqn":
                expected_config["timesteps"] = entry.config.get("timesteps")
                expected_config["profile_id"] = "main-v1"
            config_mismatch = {
                key: value
                for key, value in expected_config.items()
                if resolved_value(key) != value
            }
            if config_mismatch:
                raise CampaignArtifactError(
                    f"Campaign artifact scientific config is invalid for "
                    f"{entry.logical_id}: {config_mismatch}"
                )
        if entry.config.get("requires_fresh_main") is True:
            if not self.state_path.is_file():
                raise CampaignArtifactError(
                    f"{entry.logical_id} is not bound to the fresh campaign MAIN"
                )
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            main_record = state["entries"][FRESH_MAIN_ID]
            if main_record.get("status") != "completed":
                raise CampaignArtifactError(
                    f"{entry.logical_id} is not bound to the fresh campaign MAIN"
                )
            main_dir = self._record_attempt_path(main_record)
            expected_source = {
                "source_run_id": main_dir.name,
                "source_manifest_sha256": sha256_file(
                    main_dir / "artifact_manifest.json"
                ),
            }
            source_mismatch = {
                key: value
                for key, value in expected_source.items()
                if run.get(key) != value
            }
            if source_mismatch:
                raise CampaignArtifactError(
                    f"{entry.logical_id} is not bound to the fresh campaign MAIN: "
                    f"{source_mismatch}"
                )
            config_path = attempt_dir / "config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            for key, value in expected_source.items():
                if config.get(key) != value:
                    raise CampaignArtifactError(
                        f"{entry.logical_id} config is not bound to the fresh campaign MAIN"
                    )
        if entry.config.get("requires_phase2_input") is True:
            expected_input = self.paths.phase2_input_sha256
            config = json.loads(
                (attempt_dir / "config.json").read_text(encoding="utf-8")
            )
            input_record = config.get("input")
            if (
                expected_input is None
                or not isinstance(input_record, Mapping)
                or input_record.get("sha256") != expected_input
            ):
                raise CampaignArtifactError(
                    "Phase 2 artifact is invalid: input hash is not campaign-bound"
                )
        return manifest

    def _fresh_main_dir(self, state: Mapping[str, Any]) -> Path:
        record = state["entries"][FRESH_MAIN_ID]
        if record.get("status") != "completed":
            raise CampaignDependencyError("Required fresh campaign MAIN is not completed")
        path = self._record_attempt_path(record)
        attempt = int(record["attempts"][-1]["attempt"])
        try:
            self._verify_identity(self.spec.by_id[FRESH_MAIN_ID], path, attempt=attempt)
        except CampaignArtifactError as error:
            raise CampaignDependencyError(
                "Required fresh campaign MAIN is invalid or historical substitute evidence"
            ) from error
        return path

    def _assert_dependencies(
        self,
        entry: CampaignEntry,
        state: Mapping[str, Any],
    ) -> None:
        for dependency in entry.depends_on:
            record = state["entries"][dependency]
            if record.get("status") not in {"completed", "reused"}:
                label = "fresh campaign MAIN" if dependency == FRESH_MAIN_ID else dependency
                raise CampaignDependencyError(
                    f"Selected {entry.logical_id} requires validated {label}; "
                    "unselected dependencies are not executed automatically"
                )
            if dependency == FRESH_MAIN_ID:
                self._fresh_main_dir(state)
            elif record.get("status") == "completed":
                dependency_entry = self.spec.by_id[dependency]
                dependency_path = self._record_attempt_path(record)
                attempt = int(record["attempts"][-1]["attempt"])
                try:
                    self._verify_identity(dependency_entry, dependency_path, attempt=attempt)
                except CampaignArtifactError as error:
                    raise CampaignDependencyError(
                        f"Selected {entry.logical_id} dependency is invalid: {dependency}"
                    ) from error

    def _validate_cache(self, state: dict[str, Any]) -> str:
        try:
            result = dict(self.cache_validator(self.paths.dataset_root, self.paths.cache_root))
        except Exception as error:
            raise CampaignDependencyError(f"Validated cache is required: {error}") from error
        if result.get("validation_status") != "valid":
            raise CampaignDependencyError("Validated cache is required")
        manifest_path = self.paths.cache_root / "cache_manifest.json"
        digest = result.get("manifest_sha256")
        if digest is None:
            if not manifest_path.is_file():
                raise CampaignDependencyError("Validated cache manifest is missing")
            digest = sha256_file(manifest_path)
        bound = state.get("cache_manifest_sha256")
        if bound is not None and bound != digest:
            raise CampaignDependencyError("Validated cache hash changed during the campaign")
        if bound is None:
            state["cache_manifest_sha256"] = digest
            self._write_state(state)
        return str(digest)

    def _validate_phase2_input(self) -> Path:
        input_path = self.paths.phase2_input
        expected_hash = self.paths.phase2_input_sha256
        if input_path is None or expected_hash is None:
            raise CampaignDependencyError(
                "Phase 2 input hash and validated laboratory-flow path are required"
            )
        if not input_path.is_file():
            raise CampaignDependencyError(f"Phase 2 input is missing: {input_path}")
        actual_hash = sha256_file(input_path)
        if actual_hash != expected_hash:
            raise CampaignDependencyError(
                "Phase 2 input hash does not match the validated input binding"
            )
        return input_path

    def _source_main_args(self) -> tuple[str, str]:
        if not self.state_path.is_file():
            return "<fresh-campaign-main>", "<fresh-main-manifest-sha256>"
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        try:
            run_dir = self._fresh_main_dir(state)
        except CampaignDependencyError:
            return "<fresh-campaign-main>", "<fresh-main-manifest-sha256>"
        return self._repository_relative(run_dir), sha256_file(
            run_dir / "artifact_manifest.json"
        )

    def command_for(self, logical_id: str, *, attempt: int) -> list[str]:
        entry = self.spec.by_id.get(logical_id)
        if entry is None or entry.classification == "alias":
            raise CampaignSpecError(f"No subprocess command for {logical_id}")
        attempt_dir = self.attempt_dir(logical_id, attempt)
        common_identity = [
            "--campaign-id",
            self.campaign_id,
            "--logical-run-id",
            logical_id,
            "--attempt",
            str(attempt),
        ]
        if entry.runner == "qrdqn":
            command = [
                sys.executable,
                "src/train_rl_defender.py",
                "--split-mode",
                str(entry.config["split_mode"]),
                "--split-seed",
                str(entry.config["split_seed"]),
                "--model-seed",
                str(entry.config["model_seed"]),
                "--profile",
                "main-v1",
                "--timesteps",
                str(entry.config["timesteps"]),
                "--dataset-root",
                str(self.paths.dataset_root),
                "--cache-root",
                str(self.paths.cache_root),
                "--cache-policy",
                "require",
                "--artifact-root",
                self._repository_relative(attempt_dir.parent),
                "--run-id",
                attempt_dir.name,
                *common_identity,
            ]
            if entry.config.get("train_max_rows") is not None:
                command += ["--train-max-rows", str(entry.config["train_max_rows"])]
            if entry.config.get("holdout_csv") is not None:
                command += ["--holdout-csv", str(entry.config["holdout_csv"])]
            return command
        if entry.runner == "random_forest":
            command = [
                sys.executable,
                "src/baseline_random_forest.py",
                "--split-mode",
                str(entry.config["split_mode"]),
                "--split-seed",
                str(entry.config["split_seed"]),
                "--model-seed",
                str(entry.config["model_seed"]),
                "--n-jobs",
                "-1",
                "--dataset-root",
                str(self.paths.dataset_root),
                "--cache-root",
                str(self.paths.cache_root),
                "--cache-policy",
                "require",
                "--artifact-root",
                self._repository_relative(attempt_dir.parent),
                "--run-id",
                attempt_dir.name,
                *common_identity,
            ]
            if entry.config.get("train_max_rows") is not None:
                command += ["--train-max-rows", str(entry.config["train_max_rows"])]
            if entry.config.get("holdout_csv") is not None:
                command += ["--holdout-csv", str(entry.config["holdout_csv"])]
            return command

        source_main, _source_manifest = self._source_main_args()
        if entry.runner == "main_direct_validation":
            return [
                sys.executable,
                "scripts/validate_main_direct.py",
                "--run-dir",
                source_main,
                "--artifact-root",
                self._repository_relative(attempt_dir.parent),
                "--job-id",
                attempt_dir.name,
                *common_identity,
            ]
        if entry.runner == "bootstrap_ci":
            return [
                sys.executable,
                "scripts/bootstrap_ci.py",
                "--run-dir",
                source_main,
                "--output-dir",
                self._repository_relative(attempt_dir),
                "--n-boot",
                str(entry.config["n_resamples"]),
                "--boot-seed",
                str(entry.config["bootstrap_seed"]),
                *common_identity,
            ]
        if entry.runner == "duplicate_analysis":
            return [
                sys.executable,
                "scripts/analyze_duplicates.py",
                "--run-dir",
                source_main,
                "--output-dir",
                self._repository_relative(attempt_dir),
                *common_identity,
            ]
        if entry.runner == "shuffled_label_validation":
            return [
                sys.executable,
                "src/validate_checks.py",
                "--checks",
                "B",
                "--dataset-root",
                str(self.paths.dataset_root),
                "--cache-root",
                str(self.paths.cache_root),
                "--cache-policy",
                "require",
                "--artifact-root",
                self._repository_relative(attempt_dir.parent),
                "--run-id-b",
                attempt_dir.name,
                "--timesteps-b",
                str(entry.config["timesteps"]),
                "--split-seed",
                str(entry.config["split_seed"]),
                "--model-seed",
                str(entry.config["model_seed"]),
                "--shuffled-label-seed",
                str(entry.config["shuffled_label_seed"]),
                *common_identity,
            ]
        if entry.runner == "phase2_inference":
            flows = (
                str(self.paths.phase2_input)
                if self.paths.phase2_input is not None
                else "<validated-phase2-input>"
            )
            return [
                sys.executable,
                "scripts/predict_real_traffic_v2.py",
                "--flows",
                flows,
                "--run-dir",
                source_main,
                "--artifact-root",
                self._repository_relative(attempt_dir.parent),
                "--run-id",
                attempt_dir.name,
                "--export-diagnostics",
                *common_identity,
            ]
        raise CampaignSpecError(f"Unsupported runner for {logical_id}: {entry.runner}")

    def _complete_alias(self, entry: CampaignEntry, state: dict[str, Any]) -> None:
        source_id = entry.reuse_of
        if source_id is None:
            raise CampaignSpecError(f"Alias has no source: {entry.logical_id}")
        source = state["entries"][source_id]
        if source.get("status") != "completed":
            raise CampaignDependencyError(
                f"Alias {entry.logical_id} requires validated source {source_id}"
            )
        source_entry = self.spec.by_id[source_id]
        source_dir = self._record_attempt_path(source)
        source_attempt = int(source["attempts"][-1]["attempt"])
        self._verify_identity(source_entry, source_dir, attempt=source_attempt)
        record = state["entries"][entry.logical_id]
        record.update(
            {
                "status": "reused",
                "reuse_of": source_id,
                "source_manifest_sha256": sha256_file(
                    source_dir / "artifact_manifest.json"
                ),
                "artifact_dir": self._relative_attempt(source_dir),
                "validated_at": _utc_now(),
            }
        )
        self._write_state(state)

    def _mark_invalid(
        self,
        state: dict[str, Any],
        entry: CampaignEntry,
        attempt_record: dict[str, Any],
        error: BaseException,
    ) -> None:
        attempt_record["status"] = "invalid"
        attempt_record["ended_at"] = _utc_now()
        attempt_record["error"] = str(error)
        state["entries"][entry.logical_id]["status"] = "invalid"
        self._write_state(state)

    def _export_destination(self) -> Path:
        if self.paths.snapshot_root is None:
            raise CampaignExecutionError("An external export root is required")
        return self.paths.snapshot_root

    def _export_entry(
        self,
        entry: CampaignEntry,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        if entry.classification == "alias":
            raise CampaignExecutionError("Logical aliases have no physical run to export")
        record = state["entries"][entry.logical_id]
        attempt_dir = self._record_attempt_path(record)
        try:
            result = dict(
                self.run_exporter(
                    attempt_dir,
                    self._export_destination(),
                    repository_root=self.paths.repository_root,
                )
            )
            if result.get("status") != "verified":
                raise CampaignExecutionError("Run exporter did not return verified status")
        except Exception as error:
            record["export"] = {
                "status": "failed",
                "repository_relative_run_dir": self._repository_relative(attempt_dir),
                "error": str(error),
            }
            self._write_state(state)
            raise CampaignExecutionError(
                f"Verified per-run export failed after {entry.logical_id}: {error}"
            ) from error
        record["export"] = {
            key: result[key]
            for key in (
                "status",
                "repository_relative_run_dir",
                "export_directory",
                "archive_path",
                "checksum_path",
                "manifest_path",
                "archive_sha256",
            )
            if key in result
        }
        self._write_state(state)
        return result

    def _create_final_bundle(self) -> dict[str, Any]:
        if self.paths.snapshot_root is None:
            raise CampaignExecutionError("A snapshot root is required")
        destination = self.paths.snapshot_root / f"{self.campaign_id}.tar.gz"
        try:
            result = dict(self.bundle_exporter(self.campaign_dir, destination))
        except Exception as error:
            raise CampaignExecutionError(f"Final bundle verification failed: {error}") from error
        if result.get("status") != "verified":
            raise CampaignExecutionError("Final bundle exporter did not return verified status")
        return result

    def execute(
        self,
        *,
        resume: bool = False,
        stage: str | None = None,
        logical_run_id: str | None = None,
    ) -> dict[str, Any]:
        selected = self._selected(stage=stage, logical_run_id=logical_run_id)
        self._validate_preflight_gate()
        state = self._load_state(resume=resume)
        dispatched: list[str] = []
        skipped: list[str] = []
        reused: list[str] = []

        for entry in selected:
            record = state["entries"][entry.logical_id]
            if entry.classification == "alias":
                if record.get("status") == "reused":
                    skipped.append(entry.logical_id)
                    continue
                self._complete_alias(entry, state)
                reused.append(entry.logical_id)
                continue

            if record.get("status") == "completed":
                attempt_record = record["attempts"][-1]
                attempt_dir = self._record_attempt_path(record)
                try:
                    self._verify_identity(
                        entry,
                        attempt_dir,
                        attempt=int(attempt_record["attempt"]),
                    )
                except CampaignArtifactError as error:
                    self._mark_invalid(state, entry, attempt_record, error)
                    raise
                self._export_entry(entry, state)
                skipped.append(entry.logical_id)
                continue

            if record.get("status") == "running":
                previous = record["attempts"][-1]
                previous["status"] = "interrupted"
                previous["ended_at"] = _utc_now()
                record["status"] = "interrupted"
                self._write_state(state)
            if record.get("status") in {"failed", "interrupted", "invalid"} and not resume:
                raise CampaignExecutionError(
                    f"{entry.logical_id} requires --resume for a new physical attempt"
                )

            self._assert_dependencies(entry, state)
            if entry.config.get("requires_cache") is True:
                self._validate_cache(state)
            if entry.config.get("requires_phase2_input") is True:
                self._validate_phase2_input()

            attempt = len(record["attempts"]) + 1
            attempt_dir = self.attempt_dir(entry.logical_id, attempt)
            if attempt_dir.exists():
                raise CampaignExecutionError(
                    f"Refusing to overwrite existing attempt directory: {attempt_dir}"
                )
            attempt_record = {
                "attempt": attempt,
                "physical_run_id": attempt_dir.name,
                "artifact_dir": self._relative_attempt(attempt_dir),
                "status": "running",
                "started_at": _utc_now(),
                "ended_at": None,
                "command": self.command_for(entry.logical_id, attempt=attempt),
            }
            record["attempts"].append(attempt_record)
            record["status"] = "running"
            self._write_state(state)

            result = self.dispatcher(entry, attempt_record["command"], attempt_dir)
            dispatched.append(entry.logical_id)
            if result.returncode != 0:
                attempt_record["status"] = "failed"
                attempt_record["ended_at"] = _utc_now()
                attempt_record["returncode"] = int(result.returncode)
                record["status"] = "failed"
                self._write_state(state)
                raise CampaignExecutionError(
                    f"Sequential subprocess failed for {entry.logical_id}: "
                    f"returncode={result.returncode}"
                )
            try:
                manifest = self._verify_identity(entry, attempt_dir, attempt=attempt)
            except CampaignArtifactError as error:
                self._mark_invalid(state, entry, attempt_record, error)
                raise
            attempt_record.update(
                {
                    "status": "completed",
                    "ended_at": _utc_now(),
                    "returncode": 0,
                    "manifest_sha256": sha256_file(
                        attempt_dir / "artifact_manifest.json"
                    ),
                    "verified_schema_version": manifest["schema_version"],
                }
            )
            record["status"] = "completed"
            self._write_state(state)
            self._export_entry(entry, state)

        all_complete = all(
            record["status"] in {"completed", "reused"}
            for record in state["entries"].values()
        )
        response = {
            "status": "completed" if all_complete else "selection_completed",
            "campaign_id": self.campaign_id,
            "dispatched": dispatched,
            "skipped": skipped,
            "reused": reused,
            "state_path": str(self.state_path),
        }
        if all_complete:
            response["final_bundle"] = self._create_final_bundle()
        return response


__all__ = [
    "APPROVED_ALIASES",
    "CAMPAIGN_STATE_SCHEMA_VERSION",
    "CampaignArtifactError",
    "CampaignDependencyError",
    "CampaignEntry",
    "CampaignError",
    "CampaignExecutionError",
    "CampaignPaths",
    "CampaignRunner",
    "CampaignSpec",
    "CampaignSpecError",
    "DEFAULT_CAMPAIGN_ARTIFACT_ROOT",
    "FRESH_MAIN_ID",
    "STAGE_ORDER",
    "load_campaign_spec",
    "validate_campaign_spec",
]
