# LFS and Experimental Artifact Tracking Implementation Plan

**Goal:** Make fresh checkouts clean and make reproducibility-critical run artifacts committable while ignoring only future per-row prediction CSV exports and transfer bundles.

**Architecture:** Apply a forward-only Git LFS normalization: path-scope CSV attributes, add LFS handling for TensorBoard events, remove ignores for required run evidence, and convert only the eight current raw ZIP/joblib index entries into LFS pointers. Preserve materialized file hashes and existing history, then enforce the policy with Git-backed regression tests.

**Tech Stack:** Git, Git LFS, pytest, Ruff, PowerShell, Markdown

---

### Task 1: Add artifact tracking policy regression tests

**Files:**
- Create: `tests/test_artifact_tracking_policy.py`

- [ ] **Step 1:** Create Git helpers and representative policy assertions.

```python
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
    assert _attribute("datasets/CICIDS2017/Monday-WorkingHours.pcap_ISCX.csv", "filter") == "lfs"
    assert _attribute("pcaps/archive/deprecated_lab_flows_benign.csv", "filter") == "lfs"
    assert _attribute("runs/phase2/example/predictions.csv", "filter") == "lfs"
    assert _attribute("runs/final/tensorboard_scalars/loss.csv", "filter") == "lfs"
    assert _attribute("runs/final/plots/tensorboard_scalars/loss.csv", "filter") == "lfs"
    assert _attribute("runs/final/attempt-1/system_metrics.csv", "filter") == "unspecified"


def test_existing_prediction_csvs_remain_in_the_index() -> None:
    existing = "runs/phase2/P2v2_pred_20260224_004121/predictions.csv"
    result = _git("ls-files", "--error-unmatch", existing)
    assert result.returncode == 0, result.stderr
```

- [ ] **Step 2:** Run the focused test before policy changes.

```powershell
uv run pytest tests/test_artifact_tracking_policy.py -q
```

Expected: failures show run models/events/logs/checkpoints are ignored and `system_metrics.csv` still inherits the global CSV LFS rule.

### Task 2: Apply the forward artifact policy

**Files:**
- Modify: `.gitattributes`
- Modify: `.gitignore`
- Test: `tests/test_artifact_tracking_policy.py`

- [ ] **Step 1:** Replace global CSV LFS handling with scoped existing-evidence patterns and add TensorBoard event handling.

```gitattributes
datasets/CICIDS2017/*.csv filter=lfs diff=lfs merge=lfs -text
pcaps/**/*.csv filter=lfs diff=lfs merge=lfs -text
runs/**/predictions*.csv filter=lfs diff=lfs merge=lfs -text
runs/**/tensorboard_scalars/*.csv filter=lfs diff=lfs merge=lfs -text
runs/**/plots/tensorboard_scalars/*.csv filter=lfs diff=lfs merge=lfs -text
*.zip filter=lfs diff=lfs merge=lfs -text
runs/**/events.out.tfevents.* filter=lfs diff=lfs merge=lfs -text
*.pcap filter=lfs diff=lfs merge=lfs -text
*.joblib filter=lfs diff=lfs merge=lfs -text
```

- [ ] **Step 2:** Remove ignores for final models, TensorBoard events, and checkpoints; unignore run logs; ignore future per-row prediction CSVs.

```gitignore
!runs/**/stdout.log
!runs/**/stderr.log
runs/**/predictions*.csv
```

Delete these existing rules:

```gitignore
runs/cicids2017/*/checkpoints/
runs/**/events.out.tfevents.*
runs/**/model.zip
```

- [ ] **Step 3:** Run the focused policy test.

```powershell
uv run pytest tests/test_artifact_tracking_policy.py -q
```

Expected: `4 passed`.

### Task 3: Normalize the eight inconsistent current files into LFS

**Files:**
- Normalize index entries for:
  - `models/MAIN_qrdqn_cicids2017_canonical_full_random_20260609_193655.zip`
  - `models/rf_cicids2017_canonical_20260628_024735.joblib`
  - `runs/archive/cicids2017/C02_qrdqn_cicids2017_canonical_fast_random_20260223_181122/scaler.joblib`
  - `runs/archive/cicids2017/C03_qrdqn_cicids2017_canonical_full_random_20260223_232439/scaler.joblib`
  - `runs/archive/cicids2017/MAIN_qrdqn_cicids2017_canonical_fast_random_20260609_185427/scaler.joblib`
  - `runs/archive/cicids2017/MAIN_qrdqn_cicids2017_canonical_fast_random_20260609_190202/scaler.joblib`
  - `runs/archive/cicids2017/MAIN_qrdqn_cicids2017_canonical_fast_random_20260609_191901/scaler.joblib`
  - `runs/cicids2017/MAIN_qrdqn_cicids2017_canonical_full_random_20260609_193655/scaler.joblib`

- [ ] **Step 1:** Record materialized hashes before normalization.

```powershell
$lfsPaths = @(
  'models/MAIN_qrdqn_cicids2017_canonical_full_random_20260609_193655.zip',
  'models/rf_cicids2017_canonical_20260628_024735.joblib',
  'runs/archive/cicids2017/C02_qrdqn_cicids2017_canonical_fast_random_20260223_181122/scaler.joblib',
  'runs/archive/cicids2017/C03_qrdqn_cicids2017_canonical_full_random_20260223_232439/scaler.joblib',
  'runs/archive/cicids2017/MAIN_qrdqn_cicids2017_canonical_fast_random_20260609_185427/scaler.joblib',
  'runs/archive/cicids2017/MAIN_qrdqn_cicids2017_canonical_fast_random_20260609_190202/scaler.joblib',
  'runs/archive/cicids2017/MAIN_qrdqn_cicids2017_canonical_fast_random_20260609_191901/scaler.joblib',
  'runs/cicids2017/MAIN_qrdqn_cicids2017_canonical_full_random_20260609_193655/scaler.joblib'
)
$before = @{}
$lfsPaths | ForEach-Object { $before[$_] = (Get-FileHash -Algorithm SHA256 -LiteralPath $_).Hash }
```

- [ ] **Step 2:** Normalize only the named files.

```powershell
git add --renormalize -- $lfsPaths
```

- [ ] **Step 3:** Verify materialized hashes are unchanged.

```powershell
$lfsPaths | ForEach-Object {
  $after = (Get-FileHash -Algorithm SHA256 -LiteralPath $_).Hash
  if ($before[$_] -ne $after) { throw "Materialized hash changed: $_" }
}
```

Expected: no output and exit code `0`.

- [ ] **Step 4:** Verify the normalized index and LFS object store.

```powershell
git lfs status
git lfs ls-files
git lfs fsck
```

Expected: all eight paths are LFS-managed; `git lfs fsck` reports `Git LFS fsck OK`.

### Task 4: Document the operational policy

**Files:**
- Modify: `docs/reproducibility.md`

- [ ] **Step 1:** Add a concise artifact-versioning section.

```markdown
## Experimental artifact versioning

Campaign artifact roots intended for later evidence commits must be placed under
the repository, for example `runs/final_campaign/`. Final models, retained
checkpoints, TensorBoard events, joblib files, source datasets, PCAPs, and
existing prediction CSV evidence use Git LFS. Small configs, metrics,
environment metadata, manifests, checksums, logs, monitoring, timings, and
summary CSVs use normal Git.

Future per-row `predictions*.csv` exports and downloaded `.tar.gz` bundles remain
ignored. Existing committed prediction CSVs remain tracked. External exports may
use any filesystem destination outside the repository, including a same-device
sibling, and are transfer/recovery conveniences rather than proof of independent
durability.
```

- [ ] **Step 2:** Run documentation and policy tests.

```powershell
uv run pytest tests/test_artifact_tracking_policy.py tests/test_provider_neutrality.py -q
```

Expected: `7 passed`.

### Task 5: Run complete verification and commit

**Files:**
- Verify all changed files and normalized LFS entries.

- [ ] **Step 1:** Run complete gates.

```powershell
uv lock --check
uv run pytest
uv run ruff check .
git diff --check
git lfs fsck
```

Expected: lock check exits `0`; all tests pass; Ruff reports `All checks passed!`; diff check emits no errors; LFS fsck reports success.

- [ ] **Step 2:** Scan staged changes for secrets and inspect scope.

```powershell
git status --short
git diff --cached --stat
git diff --cached --check
```

Expected: only the design/plan, attribute/ignore/docs/test files, and eight LFS-normalized entries are present.

- [ ] **Step 3:** Commit the implementation.

```powershell
git commit -m "fix: track reproducibility artifacts with LFS"
```

Expected: commit succeeds without hooks or secret-scan failures.

- [ ] **Step 4:** Validate a clean temporary checkout and remove it safely.

```powershell
$validationRoot = Join-Path ([System.IO.Path]::GetTempPath()) 'tfg-lfs-validation'
if (Test-Path -LiteralPath $validationRoot) { throw "Validation path already exists: $validationRoot" }
git worktree add --detach $validationRoot HEAD
git -C $validationRoot status --short
git -C $validationRoot lfs fsck
git worktree remove $validationRoot
```

Expected: temporary checkout status is empty; LFS fsck succeeds; worktree removal succeeds.
