from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[1]
GRAPH_DIR = REPO_ROOT / "graphify-out"
GRAPH_JSON = GRAPH_DIR / "graph.json"
NEEDS_UPDATE = GRAPH_DIR / "needs_update"

IGNORE_PREFIXES = (
    "graphify-out/",
    "runs/",
    "models/",
    ".git/",
    ".codex/",
    ".obsidian/",
)

CODE_EXTS = {".py"}
KEY_DOC_PATHS = {
    "README.md",
    "AGENTS.md",
    ".github/AGENT_CONTEXT.md",
    "docs/README.md",
    "docs/AGENT_CONTEXT.md",
    "docs/results.md",
    "docs/phase2_plan.md",
    "docs/gcp_lab.md",
    "experiments/README.md",
    "experiments/nslkdd_experiments.md",
}
SEMANTIC_EXTS = {".md", ".pdf", ".png", ".jpg", ".jpeg", ".webp", ".svg"}

STRUCTURAL_PATTERNS = (
    re.compile(r"^[+-]\s*(async\s+def|def|class)\s+\w+"),
    re.compile(r"^[+-]\s*(from\s+\S+\s+import|import\s+\S+)"),
    re.compile(r"^[+-]\s*@\w+"),
    re.compile(r"^[+-]\s*(FEATURES_CANON|CICIDS2017_TO_CANON|NSL_KDD_TO_CANON|FLOWMETER_PY_TO_CANON|NSL_KDD_COLUMNS|META_COLS|TRUTH_COLS)\b"),
    re.compile(r"^[+-]\s*parser\.add_argument\("),
)


def _run_git(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )


def _normalize(path: str) -> str:
    return path.replace("\\", "/").strip()


def _is_ignored(path: str) -> bool:
    norm = _normalize(path)
    return any(norm.startswith(prefix) for prefix in IGNORE_PREFIXES)


def _is_code(path: str) -> bool:
    return Path(path).suffix.lower() in CODE_EXTS and not _is_ignored(path)


def _is_semantic_source(path: str) -> bool:
    norm = _normalize(path)
    if _is_ignored(norm):
        return False
    if norm in KEY_DOC_PATHS:
        return True
    if Path(norm).suffix.lower() in SEMANTIC_EXTS:
        return norm.startswith(("docs/", "experiments/", "report/", ".github/"))
    return False


def _rev_exists(rev: str) -> bool:
    proc = _run_git(["rev-parse", "--verify", rev])
    return proc.returncode == 0


def _diff_range(old_rev: str | None, new_rev: str | None) -> tuple[str, str] | None:
    if old_rev and new_rev:
        return old_rev, new_rev
    if _rev_exists("HEAD~1"):
        return "HEAD~1", "HEAD"
    return None


def _parse_name_status(old_rev: str, new_rev: str) -> list[dict[str, object]]:
    proc = _run_git(["diff", "--name-status", "--find-renames", old_rev, new_rev])
    if proc.returncode != 0:
        return []

    records: list[dict[str, object]] = []
    for raw_line in proc.stdout.splitlines():
        if not raw_line.strip():
            continue
        parts = raw_line.split("\t")
        status_token = parts[0]
        status = status_token[0]
        paths = [_normalize(p) for p in parts[1:]]
        records.append({"status": status, "paths": paths})
    return records


def _has_structural_patch(path: str, old_rev: str, new_rev: str) -> bool:
    proc = _run_git(["diff", "--unified=0", old_rev, new_rev, "--", path])
    if proc.returncode != 0:
        return False

    for line in proc.stdout.splitlines():
        if not line.startswith(("+", "-")) or line.startswith(("+++", "---", "@@")):
            continue
        stripped = line[1:].strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        for pattern in STRUCTURAL_PATTERNS:
            if pattern.match(line):
                return True
    return False


def _write_needs_update(reason: str, paths: list[str]) -> None:
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        f"timestamp: {datetime.now(timezone.utc).isoformat()}",
        f"reason: {reason}",
        "paths:",
    ]
    lines.extend(f"- {p}" for p in paths[:50])
    NEEDS_UPDATE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _rebuild_code_graph() -> bool:
    try:
        from graphify.watch import _rebuild_code
    except Exception as exc:  # pragma: no cover - local hook bootstrap
        print(f"[graphify auto] Unable to import graphify.watch: {exc}")
        return False

    print("[graphify auto] Structural code change detected - rebuilding code graph.")
    return bool(_rebuild_code(REPO_ROOT))


def decide_and_run(event: str, old_rev: str | None, new_rev: str | None, branch_switch: str | None) -> int:
    if not GRAPH_JSON.exists():
        print("[graphify auto] graphify-out/graph.json not found - skipping.")
        return 0

    if event == "post-checkout" and branch_switch != "1":
        print("[graphify auto] Non-branch checkout - skipping.")
        return 0

    diff_range = _diff_range(old_rev, new_rev)
    if diff_range is None:
        print("[graphify auto] No previous revision available - skipping.")
        return 0

    old_rev, new_rev = diff_range
    records = _parse_name_status(old_rev, new_rev)
    if not records:
        print("[graphify auto] No committed changes detected - skipping.")
        return 0

    prior_needs_update = NEEDS_UPDATE.read_text(encoding="utf-8") if NEEDS_UPDATE.exists() else None
    structural_paths: list[str] = []
    semantic_paths: list[str] = []

    for record in records:
        status = record["status"]
        paths = record["paths"]
        candidate_paths = paths if status != "R" else paths
        normalized_paths = [p for p in candidate_paths if not _is_ignored(p)]

        if not normalized_paths:
            continue

        for path in normalized_paths:
            if _is_code(path):
                if status in {"A", "D", "R"}:
                    structural_paths.append(path)
                elif status == "M" and _has_structural_patch(path, old_rev, new_rev):
                    structural_paths.append(path)

            if _is_semantic_source(path):
                semantic_paths.append(path)

    structural_paths = sorted(set(structural_paths))
    semantic_paths = sorted(set(semantic_paths))

    if not structural_paths and not semantic_paths:
        print("[graphify auto] No structural graph changes detected.")
        return 0

    rebuild_ok = True
    if structural_paths:
        rebuild_ok = _rebuild_code_graph()

    if semantic_paths:
        _write_needs_update("semantic sources changed - run `graphify .` for a full refresh", semantic_paths)
        print("[graphify auto] Semantic graph sources changed - wrote graphify-out/needs_update.")
    elif prior_needs_update and structural_paths:
        NEEDS_UPDATE.write_text(prior_needs_update, encoding="utf-8")
        print("[graphify auto] Preserved existing graphify-out/needs_update flag.")

    if structural_paths and not rebuild_ok:
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Selective Graphify auto-refresh for git hooks.")
    parser.add_argument("--event", required=True, choices=["post-commit", "post-checkout", "post-merge"])
    parser.add_argument("--old-rev")
    parser.add_argument("--new-rev")
    parser.add_argument("--branch-switch")
    args = parser.parse_args()

    return decide_and_run(args.event, args.old_rev, args.new_rev, args.branch_switch)


if __name__ == "__main__":
    sys.exit(main())
