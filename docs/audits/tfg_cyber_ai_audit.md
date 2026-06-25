# TFG_CYBER_AI: Exhaustive Project Audit

## 1. Executive Summary

This audit assesses the alignment between the current codebase, documentation, and thesis report (`report/`) of the `yeaight7/TFG_CYBER_AI` repository.

**Current repository/documentation alignment:**
The repository demonstrates an exceptionally high degree of internal coherence. The canonical invariants (152-dimensional observation vector, strict missingness-mask semantics, binary PERMIT/BLOCK actions, and explicit anti-leakage policy) are rigidly enforced in code and accurately described in the methodology documentation.

**Biggest code/report mismatches:**
- **Overclaiming in the Introduction:** `report/chapters/introduction.tex` describes the empirical comparative evaluation (RL vs Baseline) as an achieved, completed contribution, whereas the rest of the documentation and the codebase correctly reflect that final results are pending.
- **Terminology Drift:** The `report/` methodology uses descriptive text for validation pipelines ("direct prediction", "shuffled-label") rather than the explicit "Check A, B, C" terminology heavily used by the codebase (`validate_checks.py`).

**Biggest reproducibility risks:**
- **Dataset Fragmentation:** The documentation guides users to place CICIDS2017 CSVs in `datasets/`, but provides no direct download links or SHA-256 hashes for the raw Kaggle/UNB release. This risks users downloading pre-cleaned variants that silently alter the training distribution.
- **Dependency Mismatch:** `pyproject.toml` is missing `optuna`, which is present in `requirements.txt`. A user relying solely on `uv sync` will face import errors if they attempt hyperparameter tuning.

**Biggest code/methodology risks:**
- **The "Fast" Preset Collapse:** The codebase's `preset="fast"` uses `max_rows=100_000`. Because `load_cicids2017.py` loads CSVs alphabetically starting with Monday (100% benign traffic), a "fast" run trains the agent on an exclusively negative class dataset. The agent inevitably learns a trivial "always PERMIT" policy, obscuring debugging and testing.

***Note:*** *This audit judges alignment at the current unfinished stage. The lack of final results or completed thesis chapters is expected and not treated as a failure.*

---

## 2. What the Project Currently Implements

*(Based exclusively on verified code and files)*

**Implemented:**
- **Fixed Canonical Schema:** 76 flow-based features.
- **Observation Vector:** 152 dimensions (76 features + 76 missingness mask values where 1=present, 0=imputed).
- **CICIDS2017 Adapter:** Robust loader with explicit drops for leakage-prone fields (IPs, Timestamp, Flow ID, direct port proxies).
- **RL Environment:** Contextual bandit disguised as an RL environment (`rl_defender_env.py`). The agent evaluates independent flows step-by-step with no temporal state transitions.
- **Reward Function:** Fixed asymmetrical cost matrix (`tp=1.5, fp=-2.0, fn=-5.0, omission=0.0`).
- **QRDQN Agent:** Stable-Baselines3 / sb3-contrib integration.
- **Validation Suite:**
  - Check A (Direct prediction vs test set)
  - Check B (Shuffled-label anti-leakage)
  - Check C (Hard CSV/day split)
  - Leave-one-exact-CSV-out cross-validation
- **Phase 2 Pipeline:** Robust offline inference script (`predict_real_traffic_v2.py`) with scaler restoration and configurable clipping.

**Planned / Aspirational (Not Implemented):**
- Real-time active blocking with `iptables` / `nftables` (explicitly scoped out of Phase 2).
- Multi-agent or adversarial RL.
- Temporal or sequence-based RL transitions.

---

## 3. `report/` Special Audit

### `report.tex`
* **Status**: **GOOD**
* **Current Content**: LaTeX master file structuring the document.
* **Diagnosis**: Accurately reflects the current thesis structure.

### `report/chapters/introduction.tex`
* **Status**: **PARTIAL / MISMATCH**
* **Current Content**: Introduces the problem, the 152-dim schema, QRDQN vs RF setup, and offline inference limits.
* **Diagnosis**: Section 1.4 ("Contribution of the Thesis") claims empirical comparison and validation results are achieved.
* **Required Changes**: Soften Section 1.4 to indicate that the *framework* for comparison is the contribution, and empirical results are pending. Remove claims implying metrics are fully populated.

### `report/chapters/objectives_and_scope.tex`
* **Status**: **GOOD**
* **Current Content**: Defines technical invariants, asymmetric reward, and explicit Phase 2 limits.
* **Diagnosis**: Perfect match with codebase. Correctly uses conditional/future tense for expected outcomes.

### `report/chapters/methodology.tex`
* **Status**: **GOOD**
* **Current Content**: Details data pipeline, canonical mapping, mask math ($m_i=1/0$), exact reward matrix, and validation ladder. Explicitly states final metrics require committed artifacts.
* **Diagnosis**: Highly accurate.
* **Required Changes**: Expand Section 4.10.2 to explicitly mention codebase terminology (`Check A`, `Check B`, `Check C`) alongside the descriptive text.

### `report/chapters/state_of_the_art.tex`
* **Status**: **GOOD**
* **Current Content**: Literature review. Places NSL-KDD as historical context and CICIDS2017 as the main benchmark. Acknowledges RF baseline artifacts are pending.
* **Diagnosis**: Extremely accurate tone; successfully avoids overclaiming benchmark results.

---

## 4. Documentation vs Code Consistency Matrix

| Claim / Location | Verified Status | Evidence | Diagnosis |
| :--- | :---: | :--- | :--- |
| **Observation Space is 152-dim** (Docs & Report) | **MATCH** | `src/canonical_schema.py` & `src/rl_defender_env.py` | 76 features + 76 masks strictly enforced. |
| **Mask = 1 (valid), 0 (missing)** (Report) | **MATCH** | `src/canonical_schema.py` | Implemented correctly via zero-imputation + mask. |
| **Anti-Leakage excludes IPs/Ports** (Docs) | **MATCH** | `src/load_cicids2017.py` | Explicit `drop` logic before feature mapping. |
| **Reward = TP:1.5, FP:-2.0, FN:-5.0** (Report) | **MATCH** | `src/rl_defender_env.py` | Default dictionary matches report exactly. |
| **Empirical Results Completed** (Introduction.tex) | **MISMATCH** | `docs/results.md` & `runs/` | The latest full metrics are pending / historical artifacts only. |
| **Phase 2 uses active blocking** (Hypothetical) | **FUTURE-PLANNED** | `predict_real_traffic_v2.py` | Only offline inference is implemented. Docs correctly acknowledge this limit. |

---

## 5. Experiment and Reproducibility Audit

**Can a fresh user reproduce the main experiments?**
**Yes.** The `docs/runpod_main_experiment.md` and `docs/reproducibility.md` guides are highly explicit. The `RUN_ID` tracking and explicitly persisted artifacts ensure a clear relationship between the codebase and runs.

**Exact commands that appear valid:**
All documented CLI commands (`--smoke`, `--preset full`, `--split-mode day`, `--checks A B C`, and the complex Phase 2 inference arguments) exactly match `argparse` setups in the codebase.

**Missing prerequisites / mismatches:**
- `pyproject.toml` lacks `optuna` (which is present in `requirements.txt`).
- No direct download URL or SHA-256 for the CICIDS2017 raw dataset.

---

## 6. Methodology and Evaluation Audit

- **Leakage Risks:** Low. `load_cicids2017.py` rigorously drops identifiers. `validate_checks.py` (Check B) provides a runtime safeguard.
- **Split Risks:** **CRITICAL**. The `preset="fast"` uses `max_rows=100_000`. Alphabetical loading starting with Monday causes the entire 100k fast-dataset to be benign. The agent learns to exclusively PERMIT, ruining fast debug iterations.
- **Metric Risks:** Classification metrics are accurate. However, the RL environment is technically a Contextual Bandit, as flows are evaluated independently without temporal state transitions.

---

## 7. File/Module-Level Findings

- `src/canonical_schema.py`: Implements exactly 76 features. Highly robust.
- `src/load_cicids2017.py`: The gatekeeper. Contains the critical `max_rows` class-imbalance bug.
- `src/rl_defender_env.py`: Well-implemented logic, but acts as a bandit rather than full sequential RL.
- `src/validate_checks.py`: Excellent implementation of A/B/C checks.
- `scripts/deprecated_predict_real_traffic.py`: Obsolete.
- `src/load_nsl_kdd.py`: Obsolete/Historical.

---

## 8. Documentation Gaps

- Missing exact download link or hash manifest for CICIDS2017 in `README.md`.
- `report/chapters/methodology.tex` needs to map methodology descriptions directly to the `Check A, B, C` CLI arguments.

---

## 9. Risk Register

| Risk | Severity | Evidence | Consequence | Action |
| :--- | :---: | :--- | :--- | :--- |
| **"Fast" Preset Class Collapse** | **High** | `load_cicids2017.py` (`max_rows=100k`) + Monday CSV alphabetical ordering. | Agent only sees BENIGN traffic and learns an "always PERMIT" policy. | Shuffle CSV read order or stratify subsampling *before* truncation. |
| **Untracked Generated Artifacts** | **Medium** | Missing global rules for `models/*.zip`, `models/*.joblib`, and TensorBoard logs in `.gitignore`. | Repository bloat if a user runs large experiments locally. | Update `.gitignore`. |
| **Dataset Version Fragmentation** | **Medium** | No exact URL/hash for raw data. | User downloads a different CICIDS2017 variant, breaking mapping or metrics. | Add dataset hashes/links. |
| **Overclaiming in Report** | **Medium** | `introduction.tex` Sec 1.4 claims empirical comparisons are achieved. | Thesis debt; fails academic honesty checks if read today. | Soften language to "proposed comparison framework". |
| **Dependency Sync Miss** | **Low** | `pyproject.toml` missing `optuna`. | `uv sync` users fail to run tuning scripts. | Add `optuna` to `pyproject.toml`. |

---

## 10. Recommended Action Plan

*(Do not implement these now. Prioritized for future passes.)*

**Critical (Code):**
1. Fix `load_cicids2017.py` so `max_rows` reads a stratified sample across all days, or explicitly warn that `--smoke` / `preset="fast"` is not balanced.

**Important (Report / Documentation):**
1. Edit `report/chapters/introduction.tex` to soften claims about completed empirical baselines.
2. Update `report/chapters/methodology.tex` to explicitly use the terms "Check A", "Check B", and "Check C".
3. Add a direct download link and SHA-256 hashes for CICIDS2017 to the `README.md`.
4. Update `.gitignore` to globally ignore `models/*.zip`, `*.joblib`, `predictions.csv`, and `events.out.tfevents.*`.

**Nice-to-have:**
1. Sync `pyproject.toml` with `requirements.txt` (add `optuna`).
2. Remove `scripts/deprecated_predict_real_traffic.py` and potentially quarantine `src/load_nsl_kdd.py`.

---

## 11. Current Alignment Verdict

**Is the current unfinished repository internally coherent?**
Yes. The codebase enforces its technical invariants beautifully. The execution of the canonical mapping and missingness masks is particularly impressive.

**Is `report/` aligned with the current implementation and intended direction?**
Mostly yes. The methodology and scope chapters correctly treat the project as an experimental Phase 1 prototype pending final Phase 2 validation. Only the Introduction steps out of bounds by claiming empirical results are already completed.

**What must be corrected soon to avoid accumulating thesis debt?**
The `max_rows` truncation issue in the fast preset. If left unfixed, it will cause immense confusion during debugging because the agent will appear to "fail" by always permitting traffic, when in reality it is acting perfectly optimally for the biased data it received.

**What can wait until after implementation/results are more mature?**
Finalizing the results section, writing the conclusions, and implementing any live-blocking logic.
