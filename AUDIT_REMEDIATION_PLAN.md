# TFG_CYBER_AI — Audit Remediation Plan & Tracker

> Companion to `REPO_AUDIT_2026-06-27.md`. This is the executable, step-by-step plan to resolve the audit findings. **It is a plan + tracker only — do not treat any box as done until verified.**

---

## Context

The audit (`REPO_AUDIT_2026-06-27.md`) found a fundamentally sound, honest project whose defensibility is undermined by a cluster of fixable issues: a leaky random-split headline metric, an undisclosed `gamma=0` (contextual-bandit) framing, an under-documented closed-lab "Phase-2" evaluation, a train/inference preprocessing asymmetry, a reproducibility layer that is *designed but not executed*, and a set of hygiene/dependency problems. None require a redesign; most require **documentation, re-evaluation, and disclosure** rather than new training.

This plan turns the audit's findings into ordered, trackable work items so the thesis can be defended honestly. It is scoped to **everything in the audit** (all severities).

### Decisions locked (from owner, 2026-06-27)
- **D-1 Scope:** cover *all* findings (Critical → Low).
- **D-2 Compute:** **limited** GPU. Prefer cheap re-evaluation of the *existing* MAIN model and CPU-side work; treat any new 3M-step retrain as an explicit, optional, gated task.
- **D-3 Phase-2 data:** `pcaps/synthetic_real_traffic.csv` is **real captured packet traffic** from a closed lab (2 machines + a Raspberry Pi) that the **owner generated themselves**. ⇒ Labels are trustworthy; the limitation is *closed-lab, operator-generated, non-adversarial → limited external validity*, and the capture→extraction→labeling pipeline is not committed. **This corrects the audit's "self-labeled synthetic / circular evaluation" wording** (findings H2/A2/A3): keep "captured", drop "independiente"/"real-world" overclaims, document provenance, fix the confusing name.
- **D-4 Claim:** thesis claims **in-distribution detection, honestly caveated**. The temporal/day-split + RF-collapse evidence is presented as the *limitations / honesty signal*, **not** the headline. ⇒ No full-budget temporal run is required for the defense.

### Decisions locked — round 2 (owner, 2026-06-27) — RESOLVED execution gates
- **D-5 — A7 dedup retrain → CONDITIONAL (decided by user):** run **A1** first (zero-compute), then decide based on the actual `test∈train` duplicate-leakage magnitude. A7 stays gated; **resurface it to the owner once A1 reports.** (This is the only owner item still open, and only conditionally.)
- **D-6 — Git history → GOING-FORWARD ONLY (decided by user):** **NO history rewrite, NO force-push.** Fix `.gitattributes` / `.gitignore` / git email config for *future* commits only. Pre-existing historical bloat and the institutional email in past commits are **accepted and documented, not scrubbed.** ⇒ G1/G3/G4/G5 must NOT use `git lfs migrate`, `git filter-repo`, or any rewrite.
- **D-7 — Phase-2 rename → YES (decided by user):** rename `synthetic_real_traffic*.csv` → `lab_capture_traffic*.csv`; update `predict_real_traffic_v2.py` docstring/run-id + all doc references (CSV is gitignored, so only references change).
- **D-8 — Datasets → KEEP CICIDS + DROP NSL-KDD (decided by user):** keep `datasets/CICIDS2017/*.csv` in LFS and add an attribution/terms note; **drop legacy NSL-KDD entirely** — untrack `datasets/nsl_kdd/**` and `models/rf_nslkdd.joblib` and `.gitignore` them (going-forward per D-6).

### Decisions locked — round 3 (owner, 2026-06-27) — post-A1
- **D-9 — NO rerun; leave MAIN as-is (decided by user):** A7/A8/A9 (any 3M-step retrain / day-split / multi-seed run) are **won't-do**. Owner rationale: offline-inference accuracy ≈ training accuracy (~0.98–0.99) on independent lab traffic, and the A1 duplication (22.3% overall / 24.86% test-in-train / 40.12% of test attacks) is **reviewed and accepted** as low enough; the headline is **not** reframed around leakage and the thesis will **not** foreground a dedup caveat. *Auditor note (for the record, non-blocking):* the A1 finding is retained as an **owner-reviewed-and-accepted risk** (not erased); "train ≈ independent-lab accuracy" is the prepared rebuttal if a tribunal probes the random split. ⇒ **A3 is softened** to at most a light, factual "in-distribution" note (no prominent leakage caveat), and any A3 wording is confirmed with the owner before editing `results.md`/thesis.
- **D-10 — Phase-2 provenance (decided by user):** the committed Phase-2 traffic is **real captured packets the operator generated themselves via console commands in a physical, isolated home lab** (closed, non-adversarial). The committed GCP/Docker generator (`lab/docker/`, `gen_traffic.py`) was a **deprecated earlier attempt whose flows were unusable** — not the source of the committed data. Labels = operator intent (trustworthy). ⇒ The "**synthetic**" terminology in the docs is **inaccurate** and is corrected to "**real captured / operator-generated / closed home-lab / limited external validity**". Do **not** over-investigate; rename the file to `lab_capture_traffic*.csv` (D-7) in C2.

### Decisions locked — round 4 (owner, 2026-06-28)
- **D-11 — G5 (tracked PDFs) → WON'T-DO / keep tracking (decided by user):** keep `memoria/memoria.pdf` (and the other built PDFs) **tracked** in git. Owner rationale: the compiled thesis PDF must be downloadable from environments where LaTeX cannot be compiled, so tracking the canonical PDF is intentional and useful. ⇒ **G5 is won't-do**; do **not** `.gitignore` or `git rm --cached` the PDFs.

### How to use this tracker
- Status legend: `[ ]` todo · `[~]` in progress · `[x]` done & verified · `[-]` won't-do / N/A · `[?]` blocked on a decision.
- Work **top-down within a phase**; respect `Depends`. Tick a box only after its **Verify** step passes.
- Each task carries a **Compute** tag: `none` (docs/CPU/edit), `cheap` (re-eval / RF / minutes), `GPU` (a full or partial training run — gate against D-2).
- IDs are stable (`A1`, `B2`, …) so you can reference them in commits/PRs.

---

## Master tracker

### Phase 0 — Decisions & measurement (do first; unblocks the rest)
| ID | Task | Sev | Compute | Status |
|----|------|-----|---------|--------|
| A1 | Measure exact-duplicate rate & cross-split duplicate leakage (no training) | Critical | cheap | [x] |
| C1 | Document Phase-2 lab provenance (topology, capture, extraction, labeling) | High | none | [x] |
| F1 | Regenerate `uv.lock` (add `tune`/optuna); verify `uv sync --all-extras` | High | none | [x] |

### Phase 1 — Defense-critical correctness
| ID | Task | Sev | Compute | Status |
|----|------|-----|---------|--------|
| A2 | Leakage-free evaluation of the *actual* MAIN model on a day/CSV holdout | High | cheap | [-] skip (D-9: don't surface generalization gap) |
| A3 | Reframe headline as in-distribution + leakage caveat (docs + thesis) | Critical | none | [x] light — D1 reproducibility note; headline left as-is per D-9 |
| A4 | Bootstrap-CI the existing MAIN test metrics (no retrain) | Med | cheap | [x] stratified bootstrap; additive only (headline NOT reframed, no dupe caveat); end-to-end verified |
| A5 | Add operational metrics (balanced-acc, MCC, FPR/FNR) to eval | Med | cheap | [x] (PR-AUC omitted: discrete-action agent has no score) |
| B1 | Disclose `gamma=0` / contextual-bandit framing in docs | High | none | [x] |
| I1 | Disclose `gamma=0` in thesis methodology (specialize the loss) | High | none | [x] memoria/ (ES); report/ EN re-sync pending (I6) |
| C2 | Fix Phase-2 terminology + naming (captured-but-operator-generated) | High | none | [x] renamed → `lab_capture_traffic*`; 8 refs updated |
| C3 | Resolve train↔inference preprocessing skew (clipping) | High | cheap | [x] documented (reproducibility.md); centralize fn deferred |
| E1 | Align RF baseline to QRDQN protocol + persist artifacts | High | cheap | [x] code (balanced/scaled/aligned/JSON); RF re-run pending |
| D1 | Mint & commit the fixed-test reference manifest (`--write-reference`) | High | cheap | [x] minted; scaler matches MAIN |
| G1 | Route raw binaries to LFS **going-forward** (no rewrite, per D-6) | High | none | [x] .gitattributes (*.zip,*.pcap) |

### Phase 2 — Reproducibility, hygiene, deps
| ID | Task | Sev | Compute | Status |
|----|------|-----|---------|--------|
| D2 | Explicit top-of-`main()` RNG seeding | Med | none | [x] random/np/torch/cuda seeded at top of main(); smoke-verified |
| D3 | Relative paths + artifact SHA-256 in config/manifest writer | Med | none | [x] additive `checksums_sha256`+`relative_paths` in manifest; smoke-verified |
| A6 | Unify the 3 metric implementations into one shared fn (`labels=[0,1]`) | Med | none | [x] src/metrics_utils.py (train+predict+RF; validate_loco still has its own) |
| G2 | Untrack `graphify-out/`; regenerate/clear stale `fp=-1.5` node | Med | none | [x] untracked (306 files) + gitignored; semantic cache deleted (node gone) |
| G3 | Switch git email to GitHub noreply **going-forward** (no rewrite, per D-6) | Med | none | [x] local user.email→noreply; history note in reproducibility.md |
| G4 | Keep CICIDS in LFS + attribution; **drop legacy NSL-KDD** (per D-8) | Med | none | [x] CICIDS terms note (README+repro); NSL-KDD untracked+gitignored+legacy-flagged (5 docs) |
| F2 | Pin CI to Python 3.12; add uv cache; consider CPU torch for tests | Med | none | [x] ci.yml: py3.12 + UV_PYTHON + enable-cache (CPU-torch skipped: would break lock coherence) |
| F3 | Document/extend the CI verification + LFS limitation | Med | none | [x] reproducibility.md "CI scope" section (LFS hash = local only; split logic unit-tested in CI) |
| M16 | State MAIN hyperparams are hand-set, not Optuna-derived | Med | none | [x] metodologia.tex subsection + experiments callout (MAIN outside tune_hparams search space) |

### Phase 3 — Documentation alignment
| ID | Task | Sev | Compute | Status |
|----|------|-----|---------|--------|
| H1 | Fix `data-structure-...report.md:49` (C03/[512,256] stale) | Med | none | [x] 4 stale "C03=best" claims → MAIN callout + C03 relabeled pre-design probe (capped test, not comparable); grep-verified |
| H2 | Fix `results.md` "real lab traffic" vs "synthetic" contradiction | Med | none | [x] audit's L107 already fixed in C2; 1 residual "benign real traffic" (L224) aligned to closed-lab phrasing |
| H3 | Sync `GEMINI.md` (model path, stale-graph caveat) | Low | none | [x] placeholder→MAIN path + untracked/stale-graph status (needs_update caveat already present). ⚠ GEMINI.md is gitignored → local-only |
| B2 | Document missingness-mask semantic (constant=1 on CICIDS2017) | Med | none | [x] results.md observation-layout note + metodologia.tex (ES); code-verified (mask=1 post-fillna); PDF rebuilt 101pp |

### Phase 4 — Thesis chapters & wording
| ID | Task | Sev | Compute | Status |
|----|------|-----|---------|--------|
| I2 | Correct Phase-2 wording in intro/objetivos (drop "independiente"/real-world) | High | none | [x] 11 edits (intro+objetivos) + metodologia 21/298 + resultados:60 for consistency; closed-lab/limited-external-validity framing |
| I3 | Reconcile RF-baseline claims with code (or soften to "intended") | High | cheap | [x] **owner: re-run** (GPU N/A — sklearn CPU-only; strong CPU run approved); balanced/scaled `rf_..._20260628_024735` committed → claim now true+artifact-backed; results.md refreshed |
| I4 | Add Results / Discussion / Limitations / Ethics chapters | Med | none | [x] 4 chapters added + `\input`; artifact-backed RUN_IDs only; PDF 118pp; adversarially verified (all numbers match artifacts) |
| I5 | Hedge present-tense un-run protocols (learning-curve, multi-seed, LOO) | Med | none | [x] 5 hedges (learning-curve→conditional + "no artifact committed"; multi-seed→single-seed disclosure; LOO already hedged, untouched) |
| I6 | Designate `memoria/` (ES) canonical; re-sync `report/` (EN) after fixes | Low | none | [~] memoria/ designated canonical (`reproducibility.md`); EN `report/` re-sync of corrections + 4 new chapters **PENDING** |

### Phase 5 — Low-priority code & cleanup
| ID | Task | Sev | Compute | Status |
|----|------|-----|---------|--------|
| C4 | Tighten Phase-2 inference: `ImportError`-only fallback + record algo | Low | none | [x] `load_model`→`except ImportError` (corrupt `.zip` now propagates) + returns `(model,algo)`; `config["model_class"]` recorded |
| C5 | Assert `len(y_pred)==len(df)` before Phase-2 truth metrics | Low | none | [x] dual guard: early in `main()` (also protects `pd.concat` meta align) + top of `compute_truth_metrics` (unit-testable); 2 new tests |
| B3 | Remove/assert the unreachable unknown-label branch in env reward | Low | none | [x] construction-time binary-label guard + fail-fast `raise` in `_compute_reward`; test rewritten to assert the raise (per the "assert at construction OR update the test" option) |
| G5 | Untrack tracked PDFs (`memoria.pdf`, `report.pdf`, `informe.pdf`) | Low | none | [-] won't-do (D-11): keep the thesis PDF tracked so it's downloadable where LaTeX can't compile |
| G6 | Archive/relocate `deprecated_predict_real_traffic.py` | Low | none | [x] `git mv` → `scripts/archive/`; `parents[1]`→`parents[2]` (path fix at new depth) + archival banner; doc ref updated; no code imports it |
| F4 | Add `[tool.ruff]` config; sync `requirements-runpod`; comment `graphifyy` | Low | none | [x] `[tool.ruff]` line-length=100 + `[tool.ruff.lint]` select pinned (default set, `ruff check .` clean); `graphifyy` PyPI-name comment; runpod scope documented (dev/tune intentionally excluded) |

### Phase 6 — Optional, compute-gated (only if D-2 budget allows)
| ID | Task | Sev | Compute | Status |
|----|------|-----|---------|--------|
| A7 | Retrain MAIN on a **deduplicated** split (clean headline) | High | GPU | [-] won't-do (D-9: no rerun) |
| A8 | Full-budget day-split MAIN run (strengthens limitations evidence) | Med | GPU | [-] won't-do (D-9) |
| A9 | ≥3-seed MAIN runs for true variance (supersedes A4 bootstrap) | Med | GPU | [-] won't-do (D-9) |

_(Per **D-6**, all of Workstream G is **going-forward only** — no `git lfs migrate` / `git filter-repo` / force-push. Pre-existing history is retained and documented, not rewritten.)_

### Phase 0 — execution log (2026-06-27) ✅ COMPLETE

**A1 — duplicate / cross-split leakage** (`scripts/analyze_duplicates.py`, artifact `runs/validation/duplicate_analysis_seed42.json`). Run under the MAIN-matching stack (sklearn 1.9.0 / pandas 3.0.3); the seed-42 split sizes (**train 2,264,594 / test 566,149**) are **byte-identical to MAIN `config.json`**, so this measures the exact partition the headline model used. Results (stable across sklearn 1.8→1.9):
- Overall exact duplicates (full set): **22.30%** (631,269 / 2,830,743) feature-only; 22.29% feature+label.
- **Cross-split leakage: 24.86% of the TEST set (140,758 / 566,149) are exact duplicates of a TRAIN row** — feature+label 24.85%.
- **Test *attacks*: 40.12% (44,749 / 111,529)** duplicate a train row; test benigns 21.12%.
- Per-class duplicate rate (full set): attack **39.66%**, benign 18.03%.
- ⇒ **D-5 trigger fired:** leakage is large (a quarter of the test set, 40% of test attacks, memorizable from train). **A7 (dedup retrain) is now a live recommendation — owner to decide.**

**C1 — Phase-2 provenance** — created `pcaps/README.md` (evidence + owner-TODO checklist) and added a provenance pointer to `docs/phase2_plan.md`. Flagged the **Docker-bridge-IP (172.18.0.x) vs "2 machines + Raspberry Pi"** tension, the `dst_port 9101` vs generator default mismatch, and the **uncommitted generation/labelling pipeline** — all pending owner confirmation.

**F1 — lock regenerated** — `uv lock` added `optuna==4.9.0` (+ alembic/colorlog/greenlet/mako/pyyaml/sqlalchemy), **no other version changes**; `uv lock --check` passes; `provides-extras = ["dev","tune"]`. `uv sync --all-extras` succeeded and **realigned a stale venv** (numpy 2.4.4→2.4.6, pandas 3.0.2→3.0.3, sklearn 1.8.0→1.9.0, torch 2.11.0→2.12.0+cpu — now matching MAIN `environment.json`); `import optuna` OK. **Validation:** `pytest tests/` → 23 passed; `ruff check scripts/analyze_duplicates.py` → clean.

### Phase 1 — execution log (2026-06-27) — first + second half

**Decisions D-9 (no rerun; A7/A8/A9 won't-do) and D-10 (Phase-2 = real home-lab capture) recorded.** A2 skipped and A4 deferred per D-9 (both would surface/touch the headline the owner chose to leave as-is).

- **B1** — gamma=0 disclosure (+ "how is it different from a classifier?" answer) added to `DEFENSA_TFG_SCRIPT.md` (§10) and `DEFENSA_TFG_PROGRESO.md` (§6b).
- **I1** — gamma=0 subsection added to `memoria/capitulos/metodologia.tex` (specializes the loss to `δ=r−θ`; one-step contextual-bandit framing). `report/` (EN) re-sync pending (I6).
- **C2** — Phase-2 terminology corrected ("synthetic"→"real captured / operator-generated / closed home-lab / limited external validity") and files renamed `synthetic_real_traffic*.csv` → `lab_capture_traffic*.csv`; `.gitignore` pattern added first (renamed 825 MB file confirmed still ignored); 8 references updated (README, GEMINI, phase2_plan, results, both DEFENSA, pcaps/README, pcaps/archive/README).
- **C1** — `pcaps/README.md` rewritten to the true account; `gen_traffic.py`/`lab/docker/` marked deprecated.
- **D1** — `runs/cicids2017/test_partition_reference_seed42.json` minted: counts match MAIN, `test_set_sha256=cb175377…`, nested 500k/1M subsamples verified, and **the reproduced scaler matches the committed MAIN scaler** (artifacts provably correspond to the seed-42 split). `results.md` note flipped "pending mint" → minted.
- **C3** — preprocessing asymmetry (clipping only at inference) documented in `reproducibility.md`; shared-preprocessing-fn refactor deferred.
- **A3** — light only: the D1 reproducibility note; headline wording left as-is per D-9 (no leakage caveat added).
- **A6 + A5** — new `src/metrics_utils.py` (`confusion_to_metrics`, single source of truth, `labels=[0,1]`, adds balanced-acc/MCC/FPR/FNR); wired into `train_rl_defender.evaluate_model`, `predict_real_traffic_v2.compute_truth_metrics`, and `baseline_random_forest`. `validate_leave_one_csv_out` keeps its own (equivalent) fn for now.
- **E1** — `baseline_random_forest.py` rewritten: `class_weight="balanced"`, `scale=True`, day split aligned to Mon/Tue/Wed→Thu/Fri, writes per-sweep `config.json`+`metrics.json`. **RF re-run not executed** (heavy; run when ready to regenerate artifacts).
- **G1** — `.gitattributes`: `*.zip`/`*.pcap` → LFS going-forward (no history rewrite per D-6).

**Validation:** `pytest tests/` → **27 passed** (23 + 4 new `test_metrics_utils`); `ruff check` on all changed source → clean; `baseline_random_forest` import smoke OK.

**New files:** `src/metrics_utils.py`, `tests/test_metrics_utils.py`, `pcaps/README.md`, `runs/cicids2017/test_partition_reference_seed42.json`. **Renamed (gitignored):** the two Phase-2 CSVs. Nothing committed.

**Remaining in Phase 1:** A2 (skipped). A4 resolved below (owner approved 2026-06-27).

### Phase 1 (A4) + Phase 2 first-half — execution log (2026-06-27)

**Owner approved A4** ("A4 then Phase 2"). A4 is **additive only**: the headline is NOT reframed and NO duplicate caveat is surfaced (consistent with D-9).

- **A4** — new `scripts/bootstrap_ci.py` + `tests/test_bootstrap_ci.py` (6 tests). Recovers the exact MAIN confusion cells `(tn,fp,fn,tp)=(451631,2989,518,111011)` from the full-precision `metrics.json` + class totals, self-validates against every published metric (tol 1e-9), then **stratified** percentile-bootstraps (per-class binomial, conditioning on the fixed seed-42 class totals; `--unstratified` = unconditional multinomial). 95% CIs are tight (±0.0002–0.001): recall_attack 0.99536 `[0.99495, 0.99575]`, FPR 0.00658 `[0.00634, 0.00681]`, accuracy 0.99381 `[0.99360, 0.99401]`. `--from-model` re-ran the **saved MAIN model** over the reproduced split and regenerated the identical confusion matrix (logs model sha256 + n_test). Artifact `runs/validation/bootstrap_ci_seed42.json`; CI + operational-metrics table added to `results.md`, framed as test-set **sampling** precision (explicitly NOT training-seed variance). **Adversarially reviewed** (ml-experiment-reviewer → "sound-with-nits"); all nits fixed (stratified resampling, full-precision recovery wording, scope wording, model-provenance fields).
- **D2** — `random`/`np.random`/`torch`/`torch.cuda` seeded at top of `main()` (`train_rl_defender.py`); does not alter the seed-42 split (own `random_state`).
- **D3** — completion-block writer now adds `checksums_sha256` + `relative_paths` (model/scaler/percentiles/feature_names) to `artifact_manifest.json` (additive; committed MAIN absolute paths kept as informational). Verified by a fast smoke train.
- **G2** — `git rm -r --cached graphify-out/` (306 files) + blanket `graphify-out/` gitignore; deleted `graphify-out/cache/semantic/` so the fabricated `fp=-1.5` node is unreadable (only remaining hits are the audit docs describing it).
- **G3** — `git config --local user.email 132207361+yeaight7@users.noreply.github.com` (owner's own noreply, already in history); historical institutional-email exposure documented in `reproducibility.md` (accepted, not scrubbed, per D-6).
- **G4** — `git rm --cached datasets/nsl_kdd/** models/rf_nslkdd.joblib` + gitignored; CICIDS2017 attribution + **redistribution-terms** note added (README provenance + `reproducibility.md`); NSL-KDD flagged legacy/dropped in README status table, `src/load_nsl_kdd.py` docstring, `experiments/nslkdd_experiments.md`, `.github/AGENT_CONTEXT.md`.

**Validation:** `uv run pytest tests/` → **33 passed** (27 + 6 new `test_bootstrap_ci`); `uv run ruff check` on all changed source → clean; D2/D3 smoke train (`--preset fast`) produced a manifest with checksums + relative paths; A4 `--from-model` end-to-end match confirmed. **Change set:** 8 files modified, 3 new, 315 files untracked via `git rm --cached` (306 graphify-out + 8 NSL-KDD + 1 model; no history rewrite per D-6). **Nothing committed.**

### Phase 2 second-half — execution log (2026-06-27)

- **F2** — `.github/workflows/ci.yml`: `python-version: "3.12"` (matches MAIN 3.12.3), job-level `UV_PYTHON: "3.12"`, and `enable-cache: true` on `setup-uv` (caches the ~2.5 GB cu130 torch wheel across runs). The CPU-torch swap was **not** done: forcing a different torch wheel in CI would diverge from `uv.lock` and break lock coherence — documented in F3 instead.
- **F3** — new "Continuous integration: what CI does and does not verify" section in `docs/reproducibility.md`: the byte-identical seed-42 SHA-256 (`test_partition_reference_seed42.json`) is a **local** check (CICIDS CSVs are LFS, not pulled in CI), but the split/scaler/hash **logic** is already exercised in CI by `tests/test_load_cicids2017.py` (`test_nested_prefix_indices_deterministic`, `test_train_max_rows_keeps_test_set_identical`, `test_scale_true_refits_on_subsample`, `test_sha256_of_array_stable`); local verify commands documented (`verify_fixed_test_split.py [--skip-count-check]`).
- **M16** — added `\subsection{Procedencia de los hiperparámetros}` to `memoria/capitulos/metodologia.tex` and a "Hyperparameter provenance" callout to `experiments/cicids2017_qrdqn_experiments.md`: MAIN (`gamma=0.0`, `[1024,1024,512]`, `gradient_steps=20`) is a hand-set fixed profile **outside** the `tune_hparams.py` search space (`gamma∈[0.95,0.999]`, `net_arch∈{[256,128],[512,256],[256,256]}`, `gradient_steps∈{10,50,100}`), so it could not be an Optuna output. `report/` (EN) re-sync deferred to I6.

**Validation:** `uv run pytest tests/` → **33 passed**; `uv run ruff check .` → clean; `ci.yml` parses (py3.12 / cache / UV_PYTHON confirmed); `memoria/memoria.pdf` **rebuilt** with `latexmk` (exit 0, no LaTeX errors) — the M16 subsection compiles. **Change set (this half):** `ci.yml`, `docs/reproducibility.md`, `memoria/capitulos/metodologia.tex`, `experiments/cicids2017_qrdqn_experiments.md`, rebuilt `memoria/memoria.pdf`. **Nothing committed.**

**Remaining in Phase 2:** none — first + second half complete. A6 was already done. (Next phases: 3 docs alignment, 4 thesis chapters, 5 low-priority cleanup.)

### Phase 3 — documentation alignment — execution log (2026-06-28) ✅ COMPLETE

Recon-first (5 read-only agents found current state + all instances; audit line numbers predated the Phase 1–2 edits), edits applied single-writer, then adversarially reviewed (all numbers re-checked vs `metrics.json`/`config.json`, mask claim traced through code, thesis PDF rebuilt). Several audit references were already-stale (fixed earlier in C2/G2) — noted rather than re-done.

- **H1** — `docs/Personal Research/data-structure-and-canonical-schema-research-report.md`: the doc named the **C03** probe (`net_arch=[512,256]`, `accuracy=0.99859`) as the "best committed" model in 4 places (≈L49/51/61/92). Added a **MAIN canonical callout** (`MAIN_…20260609_193655`: full data, 566,149-row fixed test, 3M steps, `accuracy=0.99381`) and relabeled C03 a **superseded pre-design probe** whose `0.99859` was measured on a 100k-row capped test with distorted class mix → **not comparable** to MAIN. Differentiated by **`net_arch`** ([1024,1024,512] vs [512,256]), *not* gamma (both use `gamma=0.0` — verified in `train_rl_defender.py`). Historical narrative preserved.
- **H2** — `docs/results.md`: the audit's flagged L107 "real lab traffic" contradiction was **already corrected by C2**. Only one residual unqualified phrase remained — "benign real traffic" (the Early-v2 benign artifact) — aligned to the canonical "**benign operator-generated lab-capture traffic (real captured packets, closed home lab; limited external validity)**". Grep confirms no `synthetic`/`real-world`/`real lab traffic` left in the file.
- **H3** — `GEMINI.md`: filled the Phase-2 `predict_real_traffic_v2.py` example placeholders (`<MODEL_NAME>`/`<RUN_ID>`) with the concrete **MAIN** model/scaler/percentiles paths (verified to exist on disk; model lives under `models/`, not `runs/.../model.zip`), and softened the graphify intro to record it is **untracked/gitignored & may be stale** (G2). The `needs_update` caveat the audit asked for was **already present** (L84) — not re-added. **⚠ `GEMINI.md` is gitignored (`.gitignore:24`), so these edits are local-only and will not appear in commits/PRs.**
- **B2** — missingness-mask semantic, code-verified: `_clean_rows` does `±Inf→NaN→fillna(0)` (`src/load_cicids2017.py:155-166`) **before** `map_to_canonical` sets `mask[:,i]=(~bad)` (`src/canonical_schema.py:312-328`); since the CICIDS2017→canon mapping covers all 76 features, **the mask is constant=1 on native CICIDS2017** (encodes source-column presence, not per-value missingness; informative only for cross-domain/Phase-2). Documented in `docs/results.md` (observation-layout note under the MAIN table) and `memoria/capitulos/metodologia.tex` (ES paragraph in the "Máscara de presencia/ausencia" subsection). Used the repo's canonical "76 + 76 → 152-dimensional" phrasing (not "152 features").

**Validation:** grep checks per task pass (new wording present, all 4 stale H1 claims gone, no stale Phase-2 terms in results.md, GEMINI.md carries MAIN path + untracked + needs_update); `memoria/memoria.pdf` **rebuilt** with `latexmk` (exit 0, **101 pages**, no LaTeX errors — only benign hbox warnings) so the B2 thesis paragraph compiles. No Python changed → `pytest`/`ruff` unaffected (last green: 33 passed, Phase 2). **Change set:** 3 tracked files (`data-structure-…report.md`, `results.md`, `metodologia.tex`) + rebuilt `memoria.pdf`; `GEMINI.md` edited but gitignored (local-only). **Nothing committed.**

**Flagged for owner (out of Phase-3 scope, not done):** the same present-tense "this project *has* a graphify graph" drift G2 exposes also lives in **tracked** files — `AGENTS.md:77`, `.github/copilot-instructions.md:50`, `.agent/rules/graphify.md:3`. Since `GEMINI.md` is gitignored, the *shipping* equivalent of the H3 fix would be softening `AGENTS.md:77`. Left untouched to respect the documented Phase-3 scope; recommend a one-line follow-up.

### Phase 4 — thesis chapters & wording — execution log (2026-06-28) ✅ COMPLETE (I6 partial)

Recon-first (5 read-only agents: I2 wording, I3 RF claim, I5 un-run hedges, I4 full artifact inventory, I6 ES/EN diff), then edits applied single-writer, RF re-run, PDF rebuilt, and the new chapters adversarially verified (every number re-checked against the JSON artifacts). Two owner decisions taken mid-phase: **I3 → re-run RF** (not soften) and **I4 → draft all 4 full chapters**.

- **I2** — Phase-2 wording. Removed the "(capturado) de forma independiente" overclaim and added the closed-lab / operator-generated / non-adversarial / limited-external-validity framing across `introduccion.tex` (5 edits) and `objetivos_y_alcance.tex` (6 edits, incl. retitling "Validación externa" → "Validación complementaria"). For thesis-wide consistency also fixed the same phrasing in `metodologia.tex:21,298` and `resultados.tex:60` (surfaced by the adversarial review). Grep confirms no `independiente`/real-world Phase-2 overclaim remains.
- **I3** — RF baseline. The thesis claimed balanced/scaled/same-split/traceable RF; the code (`baseline_random_forest.py`) already matched but **E1 never re-ran it**, so the only committed artifact was the older unbalanced prototype (which *contradicted* the claim). Owner chose to **re-run** rather than soften. GPU was not applicable (sklearn RF is CPU-only; cuML/RAPIDS absent and unsupported on native Windows) → owner approved a **strong CPU run** (16 cores, `n_jobs=-1`, ~10 min). New committed artifact `rf_cicids2017_canonical_20260628_024735` (3 balanced/scaled sweeps, per-sweep `config.json`+`metrics.json`): random split acc **0.99872** / f1_attack **0.99676** (same 566,149 seed-42 test as MAIN); day split acc **0.76913** / recall_attack **0.08135** (now the **same Mon/Tue/Wed→Thu/Fri partition as Check C**); LOO-Wed recall_attack **0.00719**. The thesis claim is now true + artifact-backed (no methodology softening needed); `docs/results.md` RF section refreshed to the new run (old prototype `results_rf.txt` flagged superseded).
- **I4** — added 4 new chapters: `resultados.tex`, `discusion.tex`, `limitaciones.tex`, `etica.tex`, `\input` in `memoria.tex` between metodología and Bibliografía. **Only committed RUN_IDs are cited** (MAIN 0.99381; bootstrap CIs; Check A/B/C; A1 duplicates 22.30/24.86/40.12%; Phase-2 lab 0.991862; RF as above). No A2 holdout (skipped per D-9 — Check C carries the hard-generalization story), no training-size/multi-seed numbers (don't exist — declared as not-done). Key honest finding made explicit: RF marginally **edges** QRDQN in-distribution, but under the day-split QRDQN retains far more attack recall (0.52954 vs RF 0.08135) — the cost-sensitive formulation degrades more gracefully on the security-critical metric, while neither is deployment-ready.
- **I5** — hedged the present-tense learning-curve/training-size passage to conditional + "no se ha comprometido ningún artefacto de curva de aprendizaje" (`metodologia.tex` 270/273/275/280) and disclosed single-seed (line 336). LOO passage (line 138) was already correctly hedged and left untouched; Check A/B/C (artifact-backed) were **not** hedged.
- **I6** — designated `memoria/` (ES) canonical and recorded it in `docs/reproducibility.md`. **The English `report/` re-sync is PENDING** (Low priority): porting the gamma=0 / hyperparameter-provenance / mask-constant blocks to `report/chapters/methodology.tex`, the Phase-2 closed-lab wording across the EN chapters, and translating the 4 new chapters into EN.

**Validation:** `memoria/memoria.pdf` **rebuilt** with `latexmk` → **118 pages** (was 101), exit 0, no LaTeX errors, no undefined refs/citations, all 4 new chapters in the TOC. One build failure en route (spanish-babel `\es@sppercent` vs math-mode `\,\%` "Incompatible glue units") fixed by writing percentages in text mode. Adversarial fact-check of the 4 new chapters + I2/I5 edits → **PASS on all numbers, comparability, D-9 framing, no fabrication, Spanish, LaTeX**; one must-fix (`independiente` overclaim) fixed; `docs/results.md` RF staleness fixed. RF run log: `scratchpad/rf_run.log` (exit 0). **Change set:** `introduccion.tex`, `objetivos_y_alcance.tex`, `metodologia.tex` modified; `resultados.tex`/`discusion.tex`/`limitaciones.tex`/`etica.tex` new; `memoria.tex` (`\input`); `memoria.pdf` rebuilt; `docs/results.md` + `docs/reproducibility.md` updated; new RF artifacts under `runs/cicids2017/baseline_random_forest_comparison/rf_..._20260628_024735__*/` + `models/rf_cicids2017_canonical_20260628_024735.joblib`. **Nothing committed.**

**Remaining (Low priority):** I6 EN `report/` re-sync (3 methodology blocks + Phase-2 wording + 4-chapter translation). Optional: delete the superseded `results_rf.txt`; commit/gitignore decision for the new RF `.joblib`.

### Phase 5 first-half — execution log (2026-06-28) — C4 · C5 · B3

Recon-first (live source, not the audit's stale line numbers: `rl_defender_env.py` is under `src/`, truth-metrics call is L435 not L447). Branch `chore/yeaight7/audit-remediation-plan-phase-5` confirmed to fully contain merged `main` (Phase 1–4 via PR #33), no divergence → no rebase. Green baseline first (33 passed, ruff clean), edits applied single-writer, then adversarially reviewed by a 3-lens workflow (regression / plan-fidelity / silent-failure) — **approve-with-nits, zero must-fix**; the two surfaced nits (O(n) label check, trailing newlines) were applied.

- **C4** — `scripts/predict_real_traffic_v2.py` `load_model`: bare `except Exception` → **`except ImportError`** only, so a corrupt/incompatible `.zip` now **propagates** instead of being silently masked by the DQN fallback (DQN fallback still fires only when `sb3_contrib` is genuinely absent; `ModuleNotFoundError ⊂ ImportError` still caught). Returns `(model, algo)`; `algo = type(model).__name__` recorded as `config["model_class"]` (provenance home; metrics.json left for measurements).
- **C5** — `scripts/predict_real_traffic_v2.py`: `len(y_pred)==len(df)` enforced by **two** `ValueError` guards — one early in `main()` right after `batched_predict` (also protects the downstream `pd.concat([meta, out_df])` silent-misalignment) and one at the top of `compute_truth_metrics` (explicit contract, directly unit-testable). 2 new tests (`test_compute_truth_metrics_row_mismatch_raises`, `..._aligned_ok`).
- **B3** — `src/rl_defender_env.py`: the unreachable unknown-label `else` branch in `_compute_reward` (which silently treated unknown labels as attacks) replaced by (1) a construction-time binary-label guard in `__init__` (`np.isin(self.y, valid_labels).all()`, O(n); raises `ValueError` with offending labels) and (2) a fail-fast `raise` in the `_compute_reward` `else`. Verified safe across all 6 env construction sites (CICIDS loaders yield strictly `{0,1}` via `(labels!=benign).astype(int64)`; single-class/empty/float-cast `y` all pass). `tests/test_reward_config.py::test_unknown_label_reward` → `test_unknown_label_rejected` (now asserts the raise at both construction and `_compute_reward`); no other test covered the old silent behavior, so no coverage weakened.

**Validation:** `uv run pytest tests/` → **35 passed** (33 + 2 new C5); `uv run ruff check` on all 4 changed files → clean. No Python contract broken; no out-of-scope files touched; nothing committed (working-tree only). **Change set:** `scripts/predict_real_traffic_v2.py`, `src/rl_defender_env.py`, `tests/test_predict_real_traffic_v2.py`, `tests/test_reward_config.py`.

**Remaining in Phase 5 (second half):** G5 (untrack tracked PDFs), G6 (archive `deprecated_predict_real_traffic.py`), F4 (`[tool.ruff]` config + `requirements-runpod` sync + `graphifyy` comment).

### Phase 5 second-half — execution log (2026-06-28) — G5 (won't-do) · G6 · F4

Recon-first (confirmed no code imports the deprecated script; `requirements-runpod` is a training-only runtime; whole-repo `ruff check .` clean before any config change). Owner decision **D-11** taken: keep the thesis PDF tracked → **G5 won't-do**.

- **G5** — **won't-do (D-11).** Keep `memoria/memoria.pdf` (and other built PDFs) tracked so the compiled thesis is downloadable from environments without a LaTeX toolchain. No `.gitignore`/`git rm --cached`.
- **G6** — `git mv scripts/deprecated_predict_real_traffic.py` → **`scripts/archive/`** (per the repo's per-directory `*/archive/` convention; git records it as a rename). Fixed the in-file repo-root resolution for the new depth (`Path(__file__).resolve().parents[1]` → `parents[2]`) so the archived script still resolves `REPO` correctly, and added an archival banner docstring pointing to the `predict_real_traffic_v2.py` replacement. Verified **no code imports it**; updated both *live* references to the new path — the `docs/AGENT_CONTEXT.md` doc and the tracked `.graphifyignore` exclusion entry (`scripts/deprecated_… → scripts/archive/…`). The only remaining old-path mentions are dated audit logs (`REPO_AUDIT_2026-06-27.md`, `docs/audits/…_2026-06-25.md`), correctly left as historical record; the stale generated `graphify-out/` artifacts are gitignored and untouched. Validated with `py_compile` (not import — the module reloads ~250k CICIDS rows at import scope, the very issue that motivated archiving it) and confirmed `parents[2]` resolves to the repo root.
- **F4** — `pyproject.toml`: added `[tool.ruff]` (`line-length = 100`) + `[tool.ruff.lint]` (`select = ["E4","E7","E9","F"]`, ruff's default stable set pinned explicitly for cross-version reproducibility) — `ruff check .` stays **clean** repo-wide and ruff confirms it reads the config (`linter.line_length = 100`). Added a one-line comment that **`graphifyy` (double-y) is the PyPI distribution name; the import is `graphify`** (the `dev` extra already pinned `graphifyy==0.7.0`). For `requirements-runpod-cu130.txt`, chose the **"document its scope"** branch (not bloat the GPU training image): a header comment states it is the minimal training/eval runtime that intentionally omits the `dev` (pytest/ruff/graphifyy) and `tune` (optuna) extras, with a note to `pip install optuna==4.9.0` when running tuning on the box.

**Validation:** `uv run ruff check .` → **clean** (whole repo, under the new explicit config); `uv run pytest tests/` → **35 passed**; `uv lock --check` → resolved/coherent (pyproject dep set unchanged — only tool-config + a comment added); `pyproject.toml`/`requirements-runpod` parse; archived script `py_compile` OK and `parents[2]` → repo root verified. **Change set (this half):** `AUDIT_REMEDIATION_PLAN.md`, `docs/AGENT_CONTEXT.md`, `.graphifyignore` (path-sync to archive), `pyproject.toml`, `requirements-runpod-cu130.txt`, `scripts/deprecated_predict_real_traffic.py` → `scripts/archive/…` (rename + edits). **Nothing committed.**

**Phase 5 status:** first half (C4·C5·B3) ✅ + second half (G6·F4) ✅; G5 deliberately won't-do (D-11). Phase 5 complete.

---

## Detailed tasks

### Workstream A — Evaluation rigor & headline reframe

**A1 — Measure duplicate rate & cross-split leakage (Critical, cheap)**
- *What:* Quantify, with zero training, how badly duplicate flows inflate the headline. This evidence drives A3 and the A7 go/no-go.
- *Files:* new `scripts/analyze_duplicates.py` (read-only analysis). Reuse `load_cicids2017._prepare_cicids_features` / `load_cicids2017_binary` (it already builds `X_clean,y_clean` at `load_cicids2017.py:351`).
- *How:* Load the full canonical matrix; compute (a) # exact-duplicate `(features+label)` rows and %, (b) reproduce the seed-42 random split and count test rows that are exact duplicates of a train row (the actual leakage channel), (c) repeat per attack class.
- *Verify:* script prints duplicate % overall and the count of test∈train duplicates; commit the numbers into `docs/results.md` as evidence.
- *Depends:* —

**A2 — Leakage-free evaluation of the *actual* MAIN model (High, cheap)**
- *What:* Produce one honest generalization number for the MAIN model **without retraining**, on a day/CSV holdout, normalized with the **MAIN persisted scaler** (not a refit one — this is the subtlety that makes Check A insufficient).
- *Files:* new `scripts/eval_main_on_holdout.py`. Reuse: `load_cicids2017_csv_split` / `load_cicids2017_exact_csv_split` (used by `baseline_random_forest.py:82-104`) with `scale=False`; load `runs/cicids2017/MAIN_.../scaler.joblib` + `train_percentiles.npz` + `model.zip`; mirror the inference math in `predict_real_traffic_v2.py:178-184,404-405` and the metric block in `:285-307`.
- *How:* hold out e.g. Friday (or Wednesday LOO); `scale=False` load → apply MAIN scaler `.transform` (+ percentile clip per the C3 decision) → `QRDQN.load` → batched `deterministic=True` predict → confusion + metrics. Write a `runs/validation/VAL_main_holdout_<ts>/` artifact (config+metrics).
- *Verify:* artifact exists with attack-recall/FPR on the holdout; cross-check direction against RF day-split collapse in `results_rf.txt`.
- *Depends:* C3 (preprocessing decision), so the eval uses the canonical preprocessing.

**A3 — Reframe headline as in-distribution + leakage caveat (Critical, none)**
- *What:* Make the random-split number explicitly "in-distribution / optimistic upper bound" everywhere it appears; foreground A1's duplicate evidence and A2's holdout number as the honest generalization picture. (Per D-4, this *replaces* the need for a temporal headline.)
- *Files:* `docs/results.md` (headline table + narrative), `README.md:177-183`, `experiments/cicids2017_qrdqn_experiments.md`; thesis handled in I-stream.
- *How:* relabel the `0.9938` table as in-distribution; add a short "Why the random split is optimistic" note citing A1 + A2 + RF collapse; stop calling MAIN a "fixed test partition" (see D1).
- *Verify:* grep `results.md`/`README.md` — every headline metric carries the in-distribution caveat; no "fixed test partition" wording for the MAIN run.
- *Depends:* A1, A2, D1.

**A4 — Bootstrap-CI the MAIN test metrics (Med, cheap)** — re-evaluate the saved model on the fixed test set with N bootstrap resamples to report mean±CI for accuracy/recall_attack/FPR. No retrain. New `scripts/bootstrap_ci.py` reusing the eval path. *Verify:* CI interval reported in `results.md`. *Depends:* —. (Superseded by A9 if GPU budget appears.)

**A5 — Operational metrics in evaluation (Med, cheap)** — extend `evaluate_model` (`train_rl_defender.py:215-260`) to also emit `balanced_accuracy`, `MCC`, `PR-AUC`, `FPR`, `FNR` (data already in the confusion matrix; `validate_leave_one_csv_out.py:106-146` already computes some — reuse). Lead results tables with recall_attack + FPR. *Verify:* new keys in a fresh `metrics.json` from a smoke run; tables updated. *Depends:* A6 (shared metric fn).

**A6 — Unify metric implementations (Med, none)** — extract one `confusion_to_metrics()` (new helper, e.g. in `src/scaling_utils.py` neighbor or a new `src/metrics_utils.py`) used by `train_rl_defender.evaluate_model`, `predict_real_traffic_v2.compute_truth_metrics`, and `validate_leave_one_csv_out`. Pass `labels=[0,1]` to sklearn calls (`train_rl_defender.py:237-238` currently omits it). *Verify:* `pytest tests/`; all three call sites import the one fn; add a unit test. *Depends:* —.

**A7 / A8 / A9 (Phase 6, GPU, gated by D-2)** — dedup retrain / full-budget day run / multi-seed. Only if budget appears; each writes a normal run artifact. A7 supersedes the inflation caveat with a clean headline; A9 supersedes A4. **A7 is additionally gated on A1's leakage number (D-5): the owner decides after A1 reports.**

### Workstream B — RL framing honesty

**B1 — Disclose `gamma=0` in docs (High, none)** — add a short subsection to `docs/DEFENSA_TFG_PROGRESO.md` and a bullet to `docs/DEFENSA_TFG_SCRIPT.md`: each flow is an independent one-step decision; QRDQN with `gamma=0` is a distributional cost-sensitive contextual bandit; the distributional head still models reward-class uncertainty at the PERMIT/BLOCK boundary; the sequential machinery is intentionally inert (the dataset doesn't react). Prep the "how is this different from a classifier?" answer. *Verify:* grep `gamma`/`bandit`/`one-step` now hits the defense docs. *Depends:* —.

**B2 — Document mask semantic (Med, none)** — note in `docs/` + methodology that the 76-wide missingness mask is **constant=1 on native CICIDS2017** (mask is computed after `fillna(0)`, `canonical_schema.py:317-327`), so it encodes *source-column presence* and only becomes informative for cross-domain/lab inference; "152 features" = 76 values + 76 presence flags. *Verify:* statement present; no doc implies 152 informative features. *Depends:* —.

**B3 — Dead unknown-label branch (Low, none)** — remove or `assert label in {0,1}` in `rl_defender_env.py:132-137` (unreachable given binary labels at `load_cicids2017.py:233-234`). *Verify:* `pytest tests/test_rl_defender_env.py`. *Depends:* —.

### Workstream C — Phase-2 lab-traffic correctness *(re-scoped per D-3)*

**C1 — Document lab provenance (High, none)** — in `docs/phase2_plan.md` (+ a short `pcaps/README.md`): describe the **real** capture setup (2 hosts + Raspberry Pi, operator-generated benign + attack traffic), the capture tool (tcpdump/tshark), the flow extraction (CICFlowMeter / flowmeter-py → the `FLOWMETER_PY_TO_CANON` keys in `predict_real_traffic_v2.py:49-126`), and **how `source_label`/`truth_y` were assigned** (by which generator script/intent). Commit the capture+extraction+labeling scripts if they exist; if not scriptable, document the manual procedure precisely. *Verify:* a reader can reconstruct the dataset from the docs; the labeling source is explicit. *Depends:* —.

**C2 — Terminology + rename (High, none — rename CONFIRMED per D-7)** — clarify everywhere that Phase-2 data is **real captured packets, operator-generated in a closed lab, non-adversarial → limited external validity** (NOT real-world traffic, NOT algorithmically synthesized). **Rename** `synthetic_real_traffic*.csv` → `lab_capture_traffic*.csv` and update `predict_real_traffic_v2.py` usage docstring + run-id suffix + **all** doc references (the CSV is gitignored, so only references change). *Verify:* no doc/file implies "synthetic-generated" or "real-world"; the new name is used consistently across `results.md`, `phase2_plan.md`, DEFENSA docs, and the script. *Depends:* C1.

**C3 — Preprocessing skew (High, cheap)** — decide the canonical preprocessing and make train and inference identical. **Recommended (compute-light):** keep the model as-is (trained unclipped) and run Phase-2 / A2 **without** percentile/z-clipping to match it, OR apply the same clipping in a one-line training-eval ablation. Document the decision in `docs/reproducibility.md`. Long-term: centralize `map→(clip?)→scale→(z-clip?)` in one shared fn imported by `train_rl_defender.py:459-465` and `predict_real_traffic_v2.py:389-413`. *Verify:* a Phase-2/holdout run uses the documented canonical preprocessing; `results.md` states it. *Depends:* —. (A2 consumes this decision.)

**C4 — Inference fallback (Low, none)** — `predict_real_traffic_v2.py:164-171`: catch `ImportError` only (not bare `Exception`); record the loaded model class in `config.json`/`metrics.json`. *Verify:* `pytest tests/test_predict_real_traffic_v2.py`. *Depends:* —.

**C5 — Row-alignment guard (Low, none)** — assert `len(y_pred)==len(df)` before `compute_truth_metrics` (`predict_real_traffic_v2.py:447`). *Verify:* unit test with a mismatched-length input raises. *Depends:* —.

### Workstream D — Reproducibility infrastructure

**D1 — Mint the reference manifest (High, cheap)** — run the **existing** flag: `python scripts/verify_fixed_test_split.py --write-reference runs/cicids2017/test_partition_reference_seed42.json --check-scaler runs/cicids2017/MAIN_.../scaler.joblib` (flags confirmed at `verify_fixed_test_split.py:81-83,73-79`). Commit the manifest. Then either re-point docs to it or, if the MAIN run's own `split_metadata` lacks the hash, clearly state the manifest verifies the *reproducible split*, not retroactively the MAIN artifact. *Verify:* manifest file exists; running verify (without `--write-reference`) passes counts+sha256+scaler. *Depends:* requires the CICIDS2017 LFS CSVs locally + the MAIN environment for byte-identical hashing (note pandas 3.0.3 / sklearn 1.9.0 sensitivity).

**D2 — Explicit RNG seeding (Med, none)** — at top of `main()` (`train_rl_defender.py:360`, after `parse_args`), add `random.seed/np.random.seed/torch.manual_seed/torch.cuda.manual_seed(args.seed)` (import `random`). Matches the claim in `metodologia.tex:320`. *Verify:* `pytest tests/test_train_rl_defender_config.py`; ruff. *Depends:* —.

**D3 — Relative paths + checksums (Med, none)** — in the config/manifest writer (`train_rl_defender.py:485-566`): store repo-relative paths (or both) and add SHA-256 of `model.zip`, `scaler.joblib`, `train_percentiles.npz`, `feature_names.json` to `artifact_manifest.json`. *Verify:* a smoke run produces a manifest with relative paths + hashes; optionally a tiny test. *Depends:* —. (Applies to *future* runs; document that committed MAIN paths are RunPod-absolute and informational.)

### Workstream E — Baselines

**E1 — Align RF baseline + artifacts (High, cheap)** — `baseline_random_forest.py` has **no argparse** (all hardcoded). Set `class_weight="balanced"` (`:41/45`), use `scale=True` to match QRDQN preprocessing (`:75,85,103`), change the day split to **Mon/Tue/Wed → Thu/Fri** to match `train_rl_defender` defaults (`:83-84`), and write `config.json`+`metrics.json` per sweep (currently only `.joblib`+stdout, `:109-115`). Surface the same-split CICIDS RF-vs-RL comparison in `results.md` (separate it from NSL-KDD numbers). *Verify:* RF run emits per-sweep JSON artifacts; the day split matches RL Check C; `results.md` baseline table is CICIDS same-split. *Depends:* C3 (so RF preprocessing matches the canonical decision).

### Workstream F — Dependencies / environment / CI

**F1 — Regenerate `uv.lock` (High, none)** — run `uv lock`; confirm an `optuna==4.9.0` package entry and `provides-extras = ["dev","tune"]`; re-commit. *Verify:* `uv sync --all-extras` resolves offline; `python -c "import optuna"` works. *Depends:* —.

**F2 — CI parity (Med, none)** — `.github/workflows/ci.yml`: set `python-version: "3.12"` (`:22`, matches `environment.json` 3.12.3); add `enable-cache: true` to the `astral-sh/setup-uv` step; consider a CPU torch wheel for tests (no GPU code is exercised) to cut the ~2.5 GB cu130 download. *Verify:* CI green and faster; lock check passes. *Depends:* F1.

**F3 — CI verification scope (Med, none)** — document in `docs/reproducibility.md` that the SHA-256 split can't be verified in CI (LFS data); optionally add a small checked-in synthetic CSV + a CI step running `verify_fixed_test_split.py --skip-count-check` against a precomputed hash to exercise the split logic. *Verify:* CI step (if added) passes. *Depends:* D1.

**F4 — Dep hygiene (Low, none)** — add `[tool.ruff]` (`select`, `line-length`) to `pyproject.toml`; add `optuna`/`graphifyy` to `requirements-runpod-cu130.txt` *or* document its scope; add a one-line comment that `graphifyy` (double-y) is the real PyPI name for the `graphify` import. *Verify:* `ruff check .` clean under explicit config. *Depends:* —.

### Workstream G — Data & repo hygiene

**G1 — Route raw binaries to LFS, going-forward (High, none — per D-6 NO history rewrite)** — add `*.zip`, `*.joblib`, `*.pcap` to `.gitattributes` so FUTURE binaries become LFS pointers (omit `*.arff` — NSL-KDD is dropped in G4). For binaries you want clean at the **current tip**, `git rm --cached <f> && git add <f>` to re-store them as LFS pointers in a *new* commit (no rewrite; old blobs remain in history — accepted per D-6). Optionally `git rm --cached models/archive/** pcaps/archive/**` to untrack archives going-forward. *Verify:* re-added binaries appear in `git lfs ls-files`; working tree clean; add a `docs/` note that pre-existing history retains the raw blobs (clone size unchanged until any future rewrite). *Depends:* — (no longer batched with a rewrite).

**G2 — Untrack `graphify-out/` + fix stale node (Med, none)** — add `graphify-out/` to `.gitignore` (uncomment the `.obsidian/` line); `git rm -r --cached graphify-out/`; regenerate the graph (`graphify .`) or delete `graphify-out/cache/semantic/` so the fabricated `fp=-1.5` node can't be read. *Verify:* `git ls-files graphify-out/` empty; `git status` clean of Obsidian churn; no `fp=-1.5` node remains. *Depends:* —.

**G3 — Git email / PII, going-forward (Med, none — per D-6 NO rewrite)** — set `git config user.email <github-noreply>` for future commits; add a one-line `docs/` note recording that older commits contain the institutional email (**accepted, not scrubbed** per D-6 — no `git filter-repo`). *Verify:* new commits use the noreply address; the historical-exposure note is recorded. *Depends:* —.

**G4 — Keep CICIDS (LFS) + drop NSL-KDD, going-forward (Med, none — per D-8)** — keep `datasets/CICIDS2017/*.csv` in LFS and add a CICIDS2017 attribution/terms note (README + `docs/reproducibility.md`, official UNB URL). Drop the legacy NSL-KDD: `git rm --cached datasets/nsl_kdd/** models/rf_nslkdd.joblib` and add both to `.gitignore` (going-forward; old history retained per D-6). Flag NSL-KDD references in docs/code as removed-legacy (note `src/load_nsl_kdd.py`, `experiments/nslkdd_experiments.md`, `models/rf_nslkdd.joblib`). *Verify:* NSL-KDD untracked at tip; CICIDS attribution note committed; no doc advertises NSL-KDD load paths as current. *Depends:* —.

**G5 — Untrack PDFs — WON'T-DO (per D-11).** *Owner decision (2026-06-28):* keep the built PDFs **tracked** so the compiled thesis is downloadable where LaTeX can't be compiled. ~~`.gitignore` `memoria/*.pdf`, `report/*.pdf`, `docs/archive/*.pdf`; `git rm --cached` them.~~ Not executed. *Depends:* —.

**G6 — Deprecated predictor (Low, none)** — move `scripts/deprecated_predict_real_traffic.py` to an `archive/` dir (it reloads 250k rows at import; not referenced by code). *Verify:* grep shows no imports; quickstarts unaffected. *Depends:* —.

### Workstream H — Documentation alignment

**H1 — Stale research note (Med, none)** — add the canonical-MAIN callout to `docs/Personal Research/data-structure-and-canonical-schema-research-report.md:49` (it still names C03/[512,256] as "best committed"); mark C03 a pre-design probe. *Verify:* grep `MAIN`/`⚠️` now hits the file; no `[512,256]`-as-primary wording. *Depends:* —.

**H2 — results.md contradiction (Med, none)** — fix `docs/results.md:107` ("real lab traffic") to match the closed-lab phrasing used at `:219`; align with C2 terminology. *Verify:* one consistent description of Phase-2 data across the file. *Depends:* C2.

**H3 — GEMINI.md sync (Low, none)** — update the Phase-2 example to the MAIN model path; add the `needs_update` stale-graph caveat present in `AGENTS.md`. *Verify:* GEMINI.md matches README/AGENTS guidance. *Depends:* G2.

**M16 — Optuna provenance (Med, none)** — state in `experiments/` + methodology that MAIN hyperparameters are a **hand-set fixed profile** (gamma=0.0, [1024,1024,512] are outside the `tune_hparams.py:70-113` search space), not an Optuna output. *Verify:* claim present; no doc implies tuning produced MAIN config. *Depends:* —.

### Workstream I — Thesis (memoria/ ES canonical; report/ EN re-synced — see I6)

**I1 — Disclose `gamma=0` (High, none)** — in `memoria/capitulos/metodologia.tex` add a subsection at **line ~225** (right after the Huber-loss block, `:208-224`): state `gamma=0`, specialize `δ = r − θ(s,a)` (bootstrap term drops), name it a one-step cost-sensitive contextual bandit, and justify the distributional head at γ=0. Either specialize the existing general-γ loss in place or add the derivation. *Verify:* methodology states γ=0 and reconciles the loss; rebuild PDF compiles. *Depends:* B1 (shared wording).

**I2 — Phase-2 wording (High, none)** *(re-scoped per D-3)* — in `introduccion.tex:18,22,27,34` and `objetivos_y_alcance.tex:46,63`: keep "capturado" (accurate — it *is* captured), but **remove "de forma independiente"** (the owner generated it, so it's not independent) and any real-world/transfer overclaim; add that it is operator-generated, closed-lab, non-adversarial, limited external validity. *Verify:* no "independiente"/real-world transfer claim; closed-lab caveat present; matches C1/C2. *Depends:* C1, C2.

**I3 — RF baseline claim vs code (High, none)** — `metodologia.tex:243-251` claims balanced/scaled/same-split/artifact RF; either keep it (once E1 makes it true) or soften to "intended protocol; the committed RF is a preliminary unbalanced/unscaled prototype." *Verify:* thesis text matches the actual `baseline_random_forest.py` state. *Depends:* E1 (decide which way).

**I4 — Missing chapters (Med, none)** — add `resultados.tex`, `discusion.tex`/`limitaciones` expansion, and an ethics/closed-lab-caveat note; `\input` them in `memoria.tex:56-59`. Report only artifact-backed numbers with RUN_IDs (MAIN in-distribution; A2 holdout; Phase-2 lab; A4 CI). *Verify:* PDF builds with new chapters; every number cites a RUN_ID. *Depends:* A2, A3, A4.

**I5 — Hedge un-run protocols (Med, none)** — apply the LOO-style hedge to the learning-curve (`metodologia.tex:253-264`) and multi-seed (`:319-321`) passages: mark implemented-and-run vs implemented-not-run vs planned. *Verify:* no present-tense claim of an unrun experiment. *Depends:* —.

**I6 — Canonical language (Low, none)** — designate `memoria/` (ES) as source of truth (most recent, 99pp); after I1–I5 land, re-sync `report/` (EN) as a translation; add a one-line note in `docs/` recording which is canonical. *Verify:* EN and ES carry the same corrected claims. *Depends:* I1–I5.

---

## Suggested execution order (compute-aware)

1. **Phase 0** (A1, C1, F1) — measure + document + unbreak the lock. All cheap/none.
2. **Phase 1** correctness — the defense-critical set. A2/A3 (honest headline), B1/I1 (gamma), C1→C2 (lab provenance + rename), C3 (preprocessing), E1 (RF), D1 (manifest), G1 (LFS going-forward).
3. **Phase 2** reproducibility/hygiene/deps; **Phase 3** docs; **Phase 4** thesis chapters.
4. **Phase 5** low-priority code/cleanup.
5. **Phase 6** *only if GPU budget appears* (A7 dedup retrain → then A3/I4 upgrade from "caveated" to "clean"; A8/A9 strengthen limitations/variance).

Map to audit §11: Phase 1 ≈ "must fix before defense"; Phases 2–4 ≈ "should fix soon"; Phase 5 ≈ "nice to have"; Phase 6 = optional upgrades.

---

## Global verification

Run after each workstream (and before declaring a phase done):
- **Unit/lint:** `uv run pytest tests/` and `uv run ruff check .` (after F4, with explicit config).
- **Lock/env:** `uv sync --all-extras` resolves; `python -c "import optuna, sb3_contrib, gymnasium"`.
- **Eval scripts:** `scripts/analyze_duplicates.py`, `scripts/eval_main_on_holdout.py`, `scripts/bootstrap_ci.py`, `scripts/verify_fixed_test_split.py` run to completion and write artifacts (requires `git lfs pull` for CICIDS2017).
- **Smoke train:** `python src/train_rl_defender.py --preset fast --timesteps 25000` produces a run dir with new metrics keys (A5), relative paths + checksums (D3), and seeded determinism (D2).
- **RF:** `python src/baseline_random_forest.py` writes per-sweep `config.json`/`metrics.json` (E1).
- **Thesis:** rebuild `memoria/memoria.pdf` (use the known biber workaround: TMPDIR + sandbox-off; accented `capítulos/` path is fine) and `report/report.pdf`; confirm no broken refs and the new chapters/wording compile.
- **Hygiene:** `git status` clean of `graphify-out/` churn (G2); `git lfs ls-files` includes migrated binaries (G1); no tracked PDFs (G5).
- **Doc consistency:** grep that every headline metric carries the in-distribution caveat (A3) and that "gamma=0"/closed-lab phrasing is consistent across docs + thesis.

---

## Owner decisions — status

| Item | Decision | Status |
|------|----------|--------|
| **A7 dedup retrain** | A1 ran (24.86% test-in-train; 40.12% of test attacks). Owner reviewed → **NO rerun**, leave MAIN as-is | ✅ RESOLVED (D-9) |
| **Git history rewrite (G1/G3/G4/G5)** | Going-forward only; no rewrite / no force-push | ✅ RESOLVED (D-6) |
| **Phase-2 rename** | Rename → `lab_capture_traffic*.csv` | ✅ RESOLVED (D-7) |
| **Datasets** | Keep CICIDS2017 in LFS + attribution; drop legacy NSL-KDD | ✅ RESOLVED (D-8) |

**The only decision still open is A7**, and only *conditionally* — it is answered by running A1 (zero compute), after which the owner chooses. Everything else is decided; execution can proceed within those constraints.
