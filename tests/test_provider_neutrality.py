from __future__ import annotations

import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NEUTRAL_GUIDE = REPO_ROOT / "docs" / "gpu_experimental_environment.md"
CAMPAIGN_SPEC = REPO_ROOT / "experiments" / "final_experiment_campaign.json"

ACTIVE_NEUTRAL_DOCS = (
    REPO_ROOT / "README.md",
    REPO_ROOT / ".github" / "AGENT_CONTEXT.md",
    REPO_ROOT / "docs" / "README.md",
    REPO_ROOT / "docs" / "reproducibility.md",
    NEUTRAL_GUIDE,
)

PROVIDER_TERMS = re.compile(
    r"\b(?:runpod|gcp|google cloud|amazon web services|aws|azure|vast\.ai|"
    r"lambda labs|paperspace)\b",
    re.IGNORECASE,
)


def _without_historical_sections(markdown: str) -> str:
    kept: list[str] = []
    skipped_level: int | None = None
    for line in markdown.splitlines():
        heading = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading:
            level = len(heading.group(1))
            if skipped_level is not None and level <= skipped_level:
                skipped_level = None
            if "historical" in heading.group(2).casefold():
                skipped_level = level
        if skipped_level is None and "historical" not in line.casefold():
            kept.append(line)
    return "\n".join(kept)


def test_active_scientific_and_setup_docs_are_provider_neutral() -> None:
    violations: dict[str, list[str]] = {}
    for path in ACTIVE_NEUTRAL_DOCS:
        text = _without_historical_sections(path.read_text(encoding="utf-8"))
        matches = sorted({match.group(0) for match in PROVIDER_TERMS.finditer(text)})
        if matches:
            violations[str(path.relative_to(REPO_ROOT))] = matches

    assert not violations, violations


def test_neutral_guide_documents_exact_campaign_and_required_caveats() -> None:
    guide = NEUTRAL_GUIDE.read_text(encoding="utf-8")
    spec = json.loads(CAMPAIGN_SPEC.read_text(encoding="utf-8"))
    entries = spec["entries"]

    assert sum(entry["classification"] == "primary_model_training" for entry in entries) == 22
    assert sum(entry["classification"] == "auxiliary" for entry in entries) == 5
    assert sum(entry["classification"] == "alias" for entry in entries) == 2
    assert "22 new primary model-training executions" in guide
    assert "five auxiliary validation, analysis, and inference jobs" in guide
    assert "two aliases" in guide
    assert all(entry["logical_id"] in guide for entry in entries)

    assert "fresh campaign MAIN" in guide
    assert "historical MAIN" in guide
    assert "targeted four-holdout generalisation study" in guide
    assert "not exhaustive eight-fold leave-one-CSV-out" in guide
    assert "seed sensitivity under a fixed 1M-row / 1,324,741-timestep budget" in guide
    assert "does not estimate variance of the 3M MAIN execution" in guide


def test_documentation_paths_and_compatibility_pointer_are_valid() -> None:
    required_paths = (
        NEUTRAL_GUIDE,
        REPO_ROOT / "docs" / "phase 2" / "AGENT_CONTEXT.md",
        REPO_ROOT / "docs" / "phase 2" / "phase2_plan.md",
        REPO_ROOT / "requirements-gpu-cu130.txt",
        REPO_ROOT / "requirements-runpod-cu130.txt",
        REPO_ROOT / "scripts" / "build_cicids_cache.py",
        REPO_ROOT / "scripts" / "preflight_gpu_environment.py",
        REPO_ROOT / "scripts" / "run_campaign.py",
        REPO_ROOT / "scripts" / "export_campaign.py",
        REPO_ROOT / "scripts" / "aggregate_campaign.py",
        REPO_ROOT / "scripts" / "generate_campaign_figures.py",
    )
    assert all(path.exists() for path in required_paths)

    pointer = (REPO_ROOT / "docs" / "runpod_main_experiment.md").read_text(encoding="utf-8")
    assert pointer.startswith("# Historical compatibility pointer")
    assert "gpu_experimental_environment.md" in pointer
    assert "historical MAIN" in pointer
    assert "python src/train_rl_defender.py" not in pointer
