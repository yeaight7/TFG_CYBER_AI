# Repository Audit — TFG_CYBER_AI

**Date:** 2026-06-27 · **Scope:** read-only state audit (no changes made) · **Repo:** `github.com/yeaight7/TFG_CYBER_AI` (branch `main`)
**Method:** orchestrator scouting + direct reading of the four crux files (`train_rl_defender.py`, `rl_defender_env.py`, `load_cicids2017.py`, `predict_real_traffic_v2.py`) + 7 parallel specialist sweeps (Opus on training/inference/leakage/thesis; Sonnet on docs/security/deps) + manual cross-check. All findings carry `file:line` evidence.

> Severity is *unified* across sweeps; where two sweeps disagreed I note both. Evidence quotes are from the code/artifacts as they exist on disk today.

---

## 1. Executive summary

The repository is **substantially more mature and more honest than a typical TFG prototype**. The pipeline is real, runs end-to-end, is artifact-tracked per run, and the code contains above-average anti-leakage discipline (identifier/port columns dropped, scaler fit on train only, a shuffled-label control, a by-day split mode, an RF baseline across three split protocols). The thesis itself is unusually self-aware about its central weakness. This is a defensible project.

The maturity is **uneven**, and the risks cluster in three places:

1. **The headline result is a leaky number.** The MAIN run (`accuracy 0.9938`, `recall_attack 0.9954`) uses a *random row-level* train/test split over temporally-correlated, **non-deduplicated** CICIDS2017 flows. The repo's own evidence shows the metric collapses under realistic splits (RF attack-recall falls from 0.998 → 0.078 day-split → 0.006 leave-one-CSV-out; the only RL temporal check falls to recall 0.53). On the comparable random split a plain RandomForest *matches/beats* the RL agent (`f1 0.9971` vs `0.9845`). **This is the single biggest defensibility risk.**
2. **The "RL" is, by design, a one-step cost-sensitive classifier.** `gamma = 0.0` (hardcoded, both profiles) collapses QRDQN's bootstrap to immediate-reward quantile regression — a *contextual bandit*. The framing is legitimate **and the thesis prose half-admits it**, but the methodology still presents the full discounted-bootstrap loss without ever stating `gamma=0`, and the defense docs/README don't prepare an answer to "how is this different from a classifier?"
3. **"Phase 2 real traffic" is self-labeled synthetic data with no committed provenance.** `predict_real_traffic_v2.py` scores against the generating lab's *own* category labels (`truth_y`/`source_label`), on a 2M-row CSV (`pcaps/synthetic_real_traffic.csv`) whose generation+labeling script is **not in the repo**. The thesis chapters call it "independently captured laboratory traffic"; the project's own internal docs correctly call it synthetic. The `0.9919` Phase-2 number is therefore an in/near-distribution self-check, not external validation.

Secondary but real: a stale `uv.lock` (the `tune`/optuna extra is absent → `uv sync --all-extras` breaks), ~116 MB of raw binaries committed outside LFS + 53 MB of raw NSL-KDD, the fixed-partition/SHA-256 reproducibility mechanism is documented as if active but its reference manifest was never minted, a single-seed point estimate with no CI, a train/inference preprocessing asymmetry (clipping only at inference), and a personally-identifying institutional email baked into public git history.

**Maturity rating:** code pipeline **B+**, reproducibility infrastructure **B− (designed > executed)**, evaluation rigor **C+ (honest tools exist but the headline uses the weak protocol)**, documentation **B (accurate where maintained, with a few stale/contradictory spots)**, thesis direction **B (sound and honest, three code-vs-text mismatches to fix)**.

---

## 2. What the repository currently implements

**Training (`src/train_rl_defender.py`).**
Loads the 8 official CICIDS2017 day-CSVs (`load_cicids2017.py` → `load_cicids2017_split`), drops identifier/leakage columns (Flow ID, Timestamp, IPs, **Source/Destination Port** — the last explicitly as a label proxy, `load_cicids2017.py:120-143`), coerces numeric, maps `inf→NaN→0` (`_clean_rows`), and binarizes labels `y = (Label.upper() != "BENIGN")` (`:233-234`). With `use_canonical=True` it maps ~78 raw columns to a fixed ordered **76-feature canonical schema** and appends a **76-wide missingness mask** → **152-dim observation** (`canonical_schema.py:332`, `combined = hstack([X, mask])`; feature names `FEATURES_CANON + ["m_"+f]`). Split is stratified random 80/20 (`train_test_split(..., stratify=y_clean, random_state=42)`, `:354-360`) or by-CSV/day. Percentiles `p0.5/p99.5` are computed on `X_train[:, :76]` and a **StandardScaler is fit on train only** (`train_rl_defender.py:459-465`). A Gymnasium env (`rl_defender_env.py`) serves one flow per `step`, returns an immediate cost-matrix reward, and `terminated` only after the dataset is exhausted. QRDQN (sb3-contrib, `MlpPolicy [1024,1024,512]`, `n_quantiles 200`, `lr 5e-5`, `buffer 1e6`, `batch 2048`, **`gamma 0.0`**, `3,000,000` timesteps, `seed 42`) trains; evaluation runs `model.predict(deterministic=True)` over the scaled test matrix and writes sklearn metrics. Per-run artifacts: `config.json`, `metrics.json`, `scaler.joblib`, `train_percentiles.npz`, `feature_names.json`, `environment.json`, `artifact_manifest.json`, `model.zip`.

**Inference / Phase 2 (`scripts/predict_real_traffic_v2.py`).**
Reads a flows CSV, harmonizes time units, maps to the **same 152-d canonical schema**, applies **percentile clipping** (train bounds, first 76 dims), loads and **reuses the train-fit scaler** (`joblib.load` + `transform`, no refit — `:404-405`), applies optional **z-clipping** (`|z|≤10`), runs batched deterministic prediction, and writes `predictions.csv`, `config.json`, `metrics.json`, `diagnostics.json`. If `truth_y`/`truth_label` columns are present it computes supervised metrics against them (`:258-307`). Artifact paths are CLI args (relative) — **portable**, unlike the absolute paths inside `config.json`.

**Validation/baselines.** `validate_checks.py` (Check A direct eval, Check B shuffled-label anti-leakage, Check C day split), `validate_leave_one_csv_out.py` (leave-one-CSV-out), `baseline_random_forest.py` (RF across random/day/LOO). `tune_hparams.py` is an Optuna probe. `verify_fixed_test_split.py` checks split determinism + scaler match.

**Net characterization:** a leakage-controlled, artifact-disciplined **distributional cost-sensitive binary flow classifier**, framed as offline RL, with a synthetic-traffic robustness probe. NSL-KDD support is legacy/secondary.

---

## 3. Strong points

- **Scaler & percentiles fit on TRAIN only** — no preprocessing leakage in the headline run (`train_rl_defender.py:459-465`; verified by `test_scale_true_refits_on_subsample`).
- **Aggressive identifier/proxy-column removal**, with Destination Port dropped *specifically* as a label proxy (`load_cicids2017.py:120-143`) — above-average for CICIDS2017.
- **Label excluded from features**; `feature_names.json` has 152 entries, none containing "label" (verified).
- **Inference reuses the persisted scaler/percentiles** rather than refitting — correct calibration discipline (`predict_real_traffic_v2.py:390-405`).
- **Single source of truth for the feature contract** (`canonical_schema.map_to_canonical`) shared by train and inference → feature order cannot silently drift.
- **A real anti-leakage control was run** (Check B shuffled-label: accuracy 0.4773 < 0.5227 majority baseline) and **a hard by-day generalization test was run and honestly reported** (Check C: recall 0.53), plus an RF baseline that honestly documents the realistic-split collapse.
- **Asymmetric cost reward** correctly implemented and matches config/docs (`rl_defender_env.py:107-139`; FN −5.0 > FP −2.0).
- **Per-run environment provenance** captured (`environment.json`: Python/torch/CUDA/lib versions, RTX 3090).
- **Dataset-content hashing exists in the loader** (`test_set_sha256`, `y_test_sha256`, … `load_cicids2017.py:702-705`) and a deterministic nested-prefix train-subsample mechanism for the size benchmark.
- **Thesis is candid about the central RL critique** (`state_of_the_art.tex:148`, `metodologia.tex:196,326`: "closer to a cost-sensitive contextual decision problem than a full MDP").
- **The CICIDS2017 LFS layer is correct** (8 CSVs are genuine LFS pointers), and run checkpoints / TB events / per-run `model.zip` are correctly git-ignored.
- **Tests freeze the MAIN hyperparameter profile** (`test_main_experiment_profile_resolves_fixed_config`) preventing silent config drift.

---

## 4. Critical issues

### C1 — Headline metrics use a leaky random split over non-deduplicated, temporally-correlated flows
- **Severity:** Critical (A3) / corroborated High (A2)
- **Location:** `src/load_cicids2017.py:350-360`; MAIN `config.json` `split_mode:"random"`; `runs/cicids2017/MAIN_.../metrics.json`; `runs/cicids2017/baseline_random_forest_comparison/results_rf.txt`
- **Evidence:** All 8 day-CSVs are concatenated then split with `train_test_split(test_size=0.2, random_state=42, stratify=y_clean)` — a **row-level shuffle**. **No `drop_duplicates` anywhere in `src/` or `scripts/`.** CICIDS2017 attack episodes produce many near-identical flows; a random shuffle places siblings in both train and test. The repo's own numbers quantify the inflation: RF `f1_attack` 0.9971 (random) → 0.1446 (day) → 0.0111 (leave-Wednesday-out); RL Check C recall 0.9954 → 0.5295. On the random split RF (`acc 0.9988`) matches/beats the RL agent.
- **Why it matters:** Random-split CICIDS2017 metrics are a textbook over-optimism failure mode. Presenting `0.9938` as "the model's performance" without the temporal caveat front-and-center is the most likely tribunal takedown — *especially* because the thesis cites the very papers documenting CICIDS2017 duplicate-flow problems (`Engelen2021`, `Lanvin2023`).
- **Recommended fix:** (a) Deduplicate before splitting (or `GroupShuffleSplit` keyed on attack episode) and report counts removed; (b) **promote the by-day / leave-one-CSV-out result to co-headline status**, with the random number explicitly labeled "in-distribution / optimistic upper bound"; (c) re-run the *temporal* split at full MAIN budget (see C2/H4).

### C2 (consolidated as High) — see H-list
The other consolidated High-severity issues (gamma non-disclosure, synthetic-traffic overclaim, train/inference preprocessing skew, no leakage-free MAIN evaluation, RF baseline mismatch, raw binaries, fixed-partition disconnect, stale lock) are listed below; each is independently capable of damaging defensibility or reproducibility.

---

### High-severity issues

**H1 — `gamma=0.0` makes the agent a contextual bandit; never disclosed in the thesis while the bootstrap loss is presented as active.**
- *Location:* `train_rl_defender.py:87,109`; `config.json:"gamma":0.0`; `rl_defender_env.py:142-168`; `memoria/capitulos/metodologia.tex:208-224` (general-γ TD loss).
- *Evidence:* With `gamma=0`, `δ = r + γ·θ⁻(s',a*) − θ(s,a)` reduces to `δ = r − θ(s,a)`; the target-network/`s'` machinery the methodology describes over ~15 lines is inert. The env's multi-step episode (`max_steps=min(10000,len(X_train))`) carries no temporal credit because flows are shuffled i.i.d. and rewards are per-flow. The methodology **never states `gamma=0`** and the defense docs/README never mention `gamma`/`bandit`/`one-step` (grep: 0 hits).
- *Why it matters:* The #1 "why call it RL?" challenge. Framing is defensible only if stated.
- *Fix:* Add a methodology subsection stating `gamma=0`, specialize the loss to `δ=r−θ(s,a)`, justify why distributional quantiles still add value at γ=0 (decision-boundary reward uncertainty), and prep a defense bullet. Apply in both `memoria/` and `report/`.

**H2 — Phase-2 "real/captured laboratory traffic" is self-labeled synthetic data with no committed provenance (circular evaluation).**
- *Location:* `scripts/predict_real_traffic_v2.py:4,129,258-307`; `pcaps/synthetic_real_traffic.csv`; `runs/phase2/P2v2_pred_20260610_161231_MAIN/metrics.json`; `memoria/.../introduccion.tex:22`, `objetivos_y_alcance.tex:46`, `methodology.tex:29`; vs `docs/DEFENSA_TFG_SCRIPT.md:234`, `docs/results.md`.
- *Evidence:* Truth labels are read from a `truth_label`/`truth_y`/`source_label` column **inside the same CSV** (`:264-273`) — i.e., the generator's own traffic category. The only generator in the repo (`lab/docker/generator/gen_traffic.py`) merely opens HTTP + closed-port connections; it does not emit the 2M-row labeled CSV. So `0.9919` measures separating a scan-style signature from an HTTP signature in the distribution that *assigned* the labels. The thesis says "independently captured laboratory traffic"; internal docs correctly say "tráfico sintético etiquetado … no debe mezclarse."
- *Why it matters:* Overclaims external validity / domain-shift evidence (the stated purpose of Phase 2). If the synthetic data derives from CICIDS2017, the "domain shift" is circular.
- *Fix:* Rename script/inputs away from "real_traffic"; commit/document the capture→CICFlowMeter→labeling pipeline; correct thesis wording to "synthetic labeled traffic"; adopt the disclaimer already written in `DEFENSA_TFG_SCRIPT.md`.

**H3 — Train/inference preprocessing skew: clipping is applied ONLY at inference.** *(A2 rated this Critical; A1 High.)*
- *Location:* `train_rl_defender.py:459-465` (percentiles computed+saved, **never applied** before scaler fit) vs `predict_real_traffic_v2.py:389-413` + MAIN Phase-2 config (`clip_z=10.0`, percentiles active).
- *Evidence:* `apply_percentile_clipping`/`apply_z_clipping` are referenced **only** in `scaling_utils.py` and `predict_real_traffic_v2.py` (repo-wide grep) — never in training. The model trained on unclipped→standardized inputs but at Phase-2 receives clipped→standardized→z-clipped inputs.
- *Why it matters:* The CICIDS2017 test metric (clip-free both sides) is internally consistent, **but** the Phase-2 number is produced under a transform the model never saw, so it is not apples-to-apples with training. (Impact is bounded to the Phase-2 comparison, which is why I rate it High rather than Critical, but the "same preprocessing" contract is broken.)
- *Fix:* Centralize one preprocessing function (`map→clip→scale→z-clip`) used by both paths; either clip in training before scaler fit, or drop clipping at inference; re-run the affected side and document the canonical choice.

**H4 — No leakage-free evaluation of the *actual* MAIN model; temporal/LOO checks use a smaller re-trained proxy, and Check A/B/C predate MAIN.**
- *Location:* `validate_leave_one_csv_out.py:281,364-379`; `validate_checks.py:324-339,485-498`; `runs/validation/VAL_checks_*_20260212/13*`.
- *Evidence:* Check C / LOO instantiate a fresh QRDQN `net_arch=[512,256]`, `lr 1e-4`, **30,000** timesteps (vs MAIN `[1024,1024,512]`, 3,000,000). The only check that loads the real MAIN model (Check A) runs on the leakage-prone **random** split. The committed VAL_checks_A/B/C artifacts are dated **2026-02** (4 months pre-MAIN) on ~10,000-row test sets, not MAIN's 566,149.
- *Why it matters:* There is **no artifact** that evaluates the MAIN model on a leakage-free split. The day-split collapse (recall 0.53) cannot be cleanly attributed to distribution shift vs under-training (100× fewer steps), so neither the optimistic nor the pessimistic number is a fair generalization figure.
- *Fix:* Run `train_rl_defender.py --preset full --split-mode day --training-profile main-experiment` (full budget) and/or a load-MAIN-model + day/LOO eval reusing the persisted scaler/percentiles; commit it; re-run Check A on the MAIN model+split.

**H5 — RF baseline described as same-protocol & class-balanced; code is neither.**
- *Location:* `baseline_random_forest.py:45,75,83,104` vs `metodologia.tex:243-251`.
- *Evidence:* `class_weight=None  # note: try balanced` (text claims "ponderación de clases balanceada"); `scale=False` (QRDQN trains scaled); RF day split = train Mon–Thu / test Fri (`:83`) vs RL Check C = Mon–Wed / Thu–Fri — **different splits**; RF writes no `config.json`/`metrics.json` artifact (text claims artifact discipline).
- *Why it matters:* The same-protocol baseline is an explicit objective (objetivo específico 5); as written, the eventual RL-vs-RF comparison is confounded in exactly the way the chapter warns against.
- *Fix:* Set `class_weight="balanced"`, `scale=True`, align day partitions, persist per-run artifacts — **or** soften the text to "intended protocol" and mark the current RF as a preliminary prototype.

**H6 — ~116 MB of raw binaries committed outside LFS, plus ~53 MB raw NSL-KDD.**
- *Location:* `models/MAIN_*.zip` (31 MB), `models/rf_nslkdd.joblib` (18 MB), `models/archive/*.zip` (~37 MB), `pcaps/archive/deprecated_lab_*.pcap` (~33 MB), `datasets/nsl_kdd/*.{txt,arff}` (~53 MB). `.gitattributes` LFS rule covers **`*.csv` only**.
- *Evidence:* `git cat-file -t` confirms these are raw blobs, not LFS pointers; pack store ≈186 MB over ~677 commits.
- *Why it matters:* Every clone pulls 116+ MB of un-deltifiable binaries; hard to remove post-hoc; NSL-KDD/CICIDS public redistribution terms are unverified (see M-list).
- *Fix:* Add `*.zip`/`*.joblib`/`*.pcap`/`*.arff` (or scoped paths) to `.gitattributes`; `git lfs migrate import`; consider dropping `models/archive`, `pcaps/archive`, and legacy NSL-KDD entirely. Verify `.pcap` files contain no real payloads before deciding history rewrite.

**H7 — Fixed-partition + SHA-256 reproducibility mechanism is disconnected from the MAIN run and its reference manifest was never minted.**
- *Location:* `scripts/verify_fixed_test_split.py:42-49,184-204`; `docs/results.md:105` ("pending mint on RunPod"); `README.md:177`; MAIN `config.json` `split_metadata` (counts/ratios only, **no `test_set_sha256`**).
- *Evidence:* `runs/cicids2017/test_partition_reference_seed42.json` **does not exist**. The MAIN run predates the hashing, so its `split_metadata` has no hash. Yet `README.md:177` labels MAIN "fixed test partition" and `results.md:105` reads as if verification is active.
- *Why it matters:* The reproducibility story currently reads stronger than reality; an examiner asking "show me the SHA-256 match" has nothing to show.
- *Fix:* Either mint+commit the reference manifest (run `verify_fixed_test_split.py` on the MAIN environment) and add a CI verify step, or rewrite the docs/README to state plainly that MAIN is a single random-seed run reproducible only by re-running the loader at `seed=42` on the identical library stack.

**H8 — `uv.lock` is stale: the `tune`/`optuna` extra is absent → `uv sync --all-extras` (the CI command) is broken.**
- *Location:* `pyproject.toml:25-27`; `uv.lock` (0 occurrences of `optuna`; `tfg-cyber-ai` shows `provides-extras=["dev"]`); `.github/workflows/ci.yml`.
- *Evidence:* The lock was generated before the `tune` extra was added. `src/tune_hparams.py:18 import optuna` would not resolve from the lock.
- *Why it matters:* The lock is the reproducibility contract; CI and fresh clones diverge from what the author tested.
- *Fix:* Run `uv lock`; confirm an `optuna==4.9.0` package entry and `provides-extras=["dev","tune"]`; re-commit.

---

### Medium-severity issues

| # | Issue | Location | Why it matters |
|---|-------|----------|----------------|
| M1 | **Single seed only; no CI / variance** | all runs `seed=42`; `train_rl_defender.py:309` | RL is high-variance; `0.9938` could be a favorable seed. Examiners expect ≥3 seeds (mean±std) or a bootstrap CI on the fixed test set (cheap — re-eval only). |
| M2 | **Accuracy led as co-headline on an 80/20 imbalanced set; no balanced-acc/MCC/PR-AUC/FPR** | `metrics.json`; `evaluate_model:246-254`; `results.md:32` | Trivial always-benign = 0.803 acc. Operational NIDS metrics (recall_attack, FPR) should lead. Data to compute them exists in the confusion matrix. |
| M3 | **Missingness mask is constant = 1.0 on native CICIDS2017** | `_clean_rows` fillna(0) *before* mask; `canonical_schema.py:317-327` | 76 of 152 dims are zero-variance during MAIN training → "152 features" overstates information available. Mask only matters cross-domain/inference. Document the column-presence semantic. |
| M4 | **Global RNG seeds (np/torch/random) not set at start of `main()`** | `train_rl_defender.py` (only `QRDQN(seed=)` at `:583`, `vec_env.seed` `:577`) vs `metodologia.tex:320` | Methodology claims seeds fixed "al comienzo de cada ejecución"; code seeds only at model construction (after split + scaler). Add an explicit top-of-main seeding block. |
| M5 | **Two/three divergent metric implementations** | `train_rl_defender.py:215-260` (sklearn), `predict_real_traffic_v2.py:258-307` (hand-rolled), `validate_leave_one_csv_out.py:106-146` (third) | Risk of subtle inconsistency; `classification_report` omits `labels=[0,1]` (single-class batch could shift keys). Extract one shared confusion→metrics fn. |
| M6 | **`graphify-out/` (~304–306 files) tracked, incl. Obsidian vault; graph is stale (`needs_update`) and contains a fabricated `fp=-1.5` node** | `graphify-out/` ; `.gitignore` has the `.obsidian/` exclusion commented out | `AGENTS.md` tells agents to read the graph first; the stale node states a reward value (`fp=-1.5`) that never existed (always −1.0 or −2.0). Untrack `graphify-out/`; regenerate or delete the semantic cache. |
| M7 | **Absolute `/workspace/...` paths + no artifact checksums in `config.json`/`artifact_manifest.json`** | MAIN `config.json:108-119`, `artifact_manifest.json` | Paths don't resolve off-RunPod (informational only — scripts use relative paths, so not fatal) and no SHA-256 ties `metrics.json` to a specific `model.zip`/`scaler`. Record relative paths + artifact hashes. |
| M8 | **Institutional email + `root` author in public git history (PII/GDPR)** | commits authored `jriveroiglesias@al.uloyola.es`; some as `root` | University email is scrapeable in a public repo. Switch to the GitHub noreply email going forward; weigh a history rewrite. |
| M9 | **CICIDS2017 (2 GB LFS) + NSL-KDD redistribution terms undocumented in a public repo** | `datasets/` | UNB/CIC terms require registration/attribution and don't clearly grant redistribution. Risk of takedown. Prefer a download script + official URL, or document terms. |
| M10 | **CI Python 3.11 vs MAIN training 3.12.3** | `ci.yml:22` vs `environment.json:2` | Minor-version float/hash/parse differences could let CI pass while training behaves differently. Pin CI to 3.12. |
| M11 | **CI installs ~2.5 GB cu130 torch with no cache; never runs the real pipeline; can't verify the SHA-256** | `ci.yml` | Slow/flaky CI; the reproducibility claim isn't CI-checked (CSVs are LFS, not pulled). Add `enable-cache: true`; consider CPU torch for tests; document the LFS limitation. |
| M12 | **Thesis missing Results/Discussion/Limitations/Ethics chapters** | `memoria.tex:56-61`, `report.tex:56-61` | Expected (unfinished), but the methodology promises learning curves, LOO aggregate, multi-seed, balanced-RF — several lack artifacts. Add a results chapter tying numbers to RUN_IDs; downgrade un-run protocols to future work. |
| M13 | **Methodology describes un-run protocols in present tense** | `metodologia.tex:253-264,319-321` | Learning-curve sweep and multi-seed are described as if executed; no orchestrator/artifacts found. Apply the same hedge already used for LOO. |
| M14 | **Stale doc: `data-structure-and-canonical-schema-research-report.md:49` still names C03/`[512,256]` as "best committed artifact"** | `docs/Personal Research/...` | Missed in the 2026-06-25 fix applied to the other 3 research notes; contradicts MAIN. Add the canonical-MAIN callout. |
| M15 | **`docs/results.md:107` says Phase 2 measures "real lab traffic"; line 219 (same file) calls it synthetic** | `docs/results.md` | Internal contradiction undermining doc credibility. Make wording consistent ("synthetic labeled lab traffic"). |
| M16 | **Optuna search space cannot have produced the MAIN hyperparameters** | `tune_hparams.py:70-113` | Searches `gamma∈[0.95,0.999]`, `net_arch≤[512,256]`; MAIN uses `gamma=0.0`, `[1024,1024,512]` — outside range. If the thesis implies tuning produced MAIN config, that's unsupported. State MAIN is a hand-set fixed profile. |

### Low-severity issues
- **L1** Dead branch: unknown-label→treat-as-attack in `rl_defender_env.py:132-137` is unreachable (labels are strictly binary). Remove or assert.
- **L2** Bare `except Exception` downgrades QRDQN→DQN silently (`predict_real_traffic_v2.py:164-171`). Catch `ImportError` only; record the loaded class.
- **L3** Phase-2 truth/prediction row alignment is positional & unguarded (`:372-447`). Assert `len(y_pred)==len(df)`.
- **L4** Thesis PDFs tracked (`memoria.pdf`, `report.pdf`, `docs/archive/informe.pdf` ≈1 MB) — regenerable; embed metadata. `.gitignore` `*.pdf` and untrack.
- **L5** `requirements-runpod-cu130.txt` lacks `optuna`/`graphifyy` vs `requirements.txt` — document scope or add.
- **L6** No `[tool.ruff]` config → defaults only; one `# noqa: E402` already in tests. Add explicit `select`/`line-length`.
- **L7** `graphifyy==0.7.0` (double-y) is the real PyPI name for the `graphify` import; **not** a typosquat, but add a one-line comment to preempt reviewer alarm.
- **L8** EN `report/` lags ES `memoria/` (canonical) — designate one source of truth; fixes must land in both or the EN will re-overclaim.
- **L9** `GEMINI.md` (gitignored, present) diverged from README (placeholder model path, stale graph guidance).
- **L10** `deprecated_predict_real_traffic.py` retained (reconstructs scaler by reloading 250k rows at import) — fine but move to an archive dir to avoid confusion.

---

## 5. Documentation / report alignment issues

**Where docs are strong:** `docs/results.md` metric values match `metrics.json` exactly (all 7 metrics + Phase-2); 3/4 Personal-Research notes carry the corrected MAIN callout; defense docs label Phase-2 data synthetic; the deprecated script is excluded from quickstarts.

**Mismatches to fix (code/artifact = source of truth):**
1. **`gamma=0` absent from README, `results.md`, both DEFENSA docs, and `experiments/*` (grep: 0 hits)** while it's the key methodological fact — H1.
2. **Thesis: "captured laboratory traffic"** vs reality "synthetic, self-labeled" — H2. The thesis chapters overclaim where the internal docs are honest.
3. **`README.md:177` "fixed test partition"** for a `split_mode:random` run with no committed partition/hash — H7.
4. **`results.md:107` "real lab traffic" vs `:219` "synthetic"** — internal contradiction — M15.
5. **`data-structure-...-report.md:49`** still elevates C03/`[512,256]` — M14.
6. **Check A/B/C presented as validating "the model"** but they predate MAIN on ~10k rows — H4/M.
7. **RF baseline text vs code** (balanced/scaled/same-split/artifacts) — H5.
8. **`results.md`/`AGENT_CONTEXT.md` `test_set_sha256` "verified per run"** while the reference manifest is pending — H7.
9. **Methodology present-tense for un-run learning-curve/multi-seed protocols** — M13.
10. **`GEMINI.md` stale** (model placeholder, no `needs_update` caveat) — L9.

Commands in docs are mostly current; the main runnability trap is `uv sync --all-extras` (H8) and the LFS-data requirement for any reproduction (not always foregrounded).

---

## 6. Reproducibility checklist

| Item | Status | Evidence | Action needed |
|------|--------|----------|---------------|
| Code availability (train/infer/eval/validate/baseline) | **OK** | All entrypoints in `src/`+`scripts/`; portable repo-relative paths in code | — |
| Dataset availability | **Partial** | CICIDS2017 in LFS (works for clones with LFS); redistribution terms unverified; NSL-KDD raw | Document official UNB URL + download script; verify license |
| Split reproducibility (MAIN) | **Partial** | Deterministic `random_state=42` + `test_set_sha256` in loader, **but** MAIN's own `config.json` has no hash; reference manifest absent; depends on pandas 3.0.3 / sklearn 1.9.0 row ordering | Mint+commit `test_partition_reference_seed42.json`; pin lib versions in docs |
| Seed capture | **Partial** | `seed=42` recorded; but np/torch/random not seeded at start of `main()` (contradicts `metodologia.tex:320`) | Add explicit top-of-main seeding block |
| Hyperparameter capture | **OK** | Full set in `config.json`; frozen by `test_main_experiment_profile_resolves_fixed_config` | — |
| Hyperparameter *provenance* | **Unclear** | MAIN values outside Optuna search space | State MAIN is a hand-set fixed profile |
| Environment capture | **OK** | `environment.json` (Python/torch/CUDA/libs/GPU) | — |
| Dependency lock coherence | **Missing** | `uv.lock` lacks `optuna`/`tune` extra → `uv sync --all-extras` breaks | `uv lock`; recommit |
| Artifact integrity (checksums) | **Missing** | `artifact_manifest.json` paths-only, no SHA-256 for model/scaler/percentiles | Add artifact hashes + relative paths |
| Model + preprocessing bundled | **OK** | `scaler.joblib`+`train_percentiles.npz`+`feature_names.json` saved per run; inference reuses them | — |
| Train↔inference preprocessing identical | **Missing/Partial** | Clipping applied only at inference (H3) | Unify preprocessing; re-run affected side |
| Leakage-free eval of the MAIN model | **Missing** | Check A on random split; Check C/LOO use a 30k-step proxy; all VAL_checks predate MAIN | Full-budget day/LOO run loading the MAIN model |
| Multiple seeds / CI on headline | **Missing** | All `seed=42`; no variance reported | ≥3 seeds or bootstrap CI on fixed test set |
| Documented runnable commands | **Partial** | Quickstarts present; `--all-extras` trap; LFS prerequisite under-stated | Fix lock; foreground `git lfs pull` |
| Hardware documented | **OK** | RTX 3090 / CUDA 13.0 in `environment.json` + `reproducibility.md`; CPU fallback path exists in code | Note CPU path viability for GPU-less reviewers |
| CI verifies the pipeline | **Partial** | Unit tests + schema check + ruff; no end-to-end / no SHA-256 verification; Python 3.11≠3.12.3 | Pin Python 3.12; add cache; document LFS limitation |
| Phase-2 synthetic data provenance | **Missing** | Generator CSV+labels not committed; `gen_traffic.py` insufficient | Commit/document generation+labeling pipeline |

---

## 7. Experiment and metrics audit

**Internal consistency:** the reported numbers are *self-consistent and artifact-backed* — `docs/results.md` matches `metrics.json` exactly (MAIN: acc 0.99381, prec_attack 0.97378, recall_attack 0.99536, f1_attack 0.98445; Phase-2: acc 0.991862, block_rate 0.252364, f1_attack 0.983801). Definitions are coherent: accuracy/precision/recall/F1 per class via sklearn (train), hand-rolled tp/tn/fp/fn (Phase-2), block_rate=`mean(pred==1)`, z-diagnostics on the first 76 scaled dims (`z_abs_max`, `z_gt10_count`). The Phase-2 MAIN run reports `z_gt10_count=0` — consistent with `clip_z=10` being active.

**Defensibility:** the metrics are **defensible-with-caveats, not as-is.** The dominant problem is *protocol*, not arithmetic:
- The headline uses the **leaky random split** (C1); the repo's own RF/Check-C evidence shows the realistic-split collapse.
- The **RL agent shows no demonstrated advantage over RandomForest** on the comparable random split (RF f1 0.9971 ≥ RL 0.9845) — a tribunal will ask "what did RL buy you?" The honest answer must rest on the *cost-sensitivity* of the decision (FPR/FNR trade-off under the reward matrix), not raw accuracy.
- **Accuracy is the wrong headline** on an 80/20 split (M2); lead with recall_attack + FPR.
- **Single seed, no CI** (M1) — can't claim the model "achieves" 0.9938.
- **Phase-2 0.9919** is an in/near-distribution self-labeled check (H2), not external validation, and is computed under a different preprocessing path (H3).
- The validation ladder is **well-designed but not yet pointed at the MAIN model** (H4).

**Leakage assessment:** preprocessing leakage is *avoided* (scaler/percentiles train-only; constant-fill is split-order-independent; identifiers/ports dropped; Label excluded). The residual leakage is **distributional** (duplicate flows across a random split), which is real and quantified by the repo itself.

---

## 8. Data and artifact hygiene

- **LFS:** correct for the 8 CICIDS2017 CSVs (genuine pointers). **Incorrect/missing** for `*.zip` (31 MB MAIN model + ~37 MB archive), `*.joblib` (18 MB), `*.pcap` (~33 MB), `*.arff`/`*.txt` (~53 MB NSL-KDD) — all raw blobs (H6). Pack ≈186 MB.
- **`.gitignore` vs reality:** correctly ignores checkpoints/TB events/per-run `model.zip`/synthetic CSVs; **but** leaves tracked: `graphify-out/` (~304–306 files incl. Obsidian vault, with the `.obsidian/` exclusion commented out — M6), `report.pdf`/`memoria.pdf`/`informe.pdf` (L4), and the raw binaries above. Live untracked `.obsidian/*.json` edits leak local view state.
- **Obsolete/duplicate:** `models/archive/*`, `runs/archive/*`, `pcaps/archive/*` (deprecated lab pcaps + 132-byte placeholder CSVs), `docs/archive/informe.*`, the dual EN/ES thesis trees, and legacy NSL-KDD. None are misleading per se (archives are labeled), but they inflate the repo and the deprecated `.pcap` files may contain real payloads (verify before any history rewrite).
- **Scaler artifacts** are small (`StandardScaler`, ~4 KB — *not* MinMaxScaler as one sweep mislabeled; verified `StandardScaler` at `train_rl_defender.py:463`), no data leakage.

---

## 9. Dependency / environment audit

- **Pins are exact and mostly consistent** across `requirements.txt`, `requirements-runpod-cu130.txt`, `pyproject.toml`, `uv.lock`, and `environment.json` (numpy 2.4.6, pandas 3.0.3, sklearn 1.9.0, gymnasium 1.2.3, sb3 2.8.0, sb3-contrib 2.8.0, joblib 1.5.3; torch 2.12.1 with a clean cpu/cu130 platform split). `environment.json` corroborates the actual MAIN run.
- **Discrepancies:** `uv.lock` is **stale** (no `optuna`/`tune` extra — H8); `requirements-runpod-cu130.txt` omits `optuna`/`graphifyy` (L5); CI Python 3.11 ≠ training 3.12.3 (M10); no `[tool.ruff]` (L6).
- **Installability:** the stack is bleeding-edge (pandas 3.x, numpy 2.4.x, torch 2.12, Python 3.12). A clean clone's reproducibility hinges on `uv.lock` being coherent — which it currently isn't (H8). The cpu/cu130 split *does* give GPU-less reviewers a CPU path; document its viability.
- **Supply chain:** `graphifyy` (double-y) is the legitimate PyPI name (import `graphify`), pinned by hash in `uv.lock` — not a typosquat, but comment it (L7). No vulnerable/conflicting/unused direct deps found; `tensorboard`/`matplotlib` correctly treated as transitive via `stable-baselines3[extra]`.
- **CI value:** tests meaningfully guard several core claims (scaler-on-train-only, split determinism on synthetic data, frozen MAIN profile, env step/reward). They do **not** exercise the real pipeline or verify the documented SHA-256 (M11).

---

## 10. Security / privacy audit

- **No live secrets** in tracked source: the only credential-like strings are clearly-templated placeholders (`NEO4J_*` in a skill doc, `MYSQL_ROOT_PASSWORD=test` in `docs/gcp_lab.md`). CI uses no secret injection. Good.
- **PII:** institutional email `jriveroiglesias@al.uloyola.es` (and a `root` author identity) is permanently in public git history (M8) — GDPR-relevant, scrapeable. Switch to GitHub noreply going forward; consider a coordinated history rewrite.
- **Machine-path leakage:** `/workspace/TFG_CYBER_AI/...` (RunPod) baked into ~22 `config.json`/`artifact_manifest.json` files (M7) — reveals infra layout, not a credential.
- **Pickle/RCE awareness:** `predict_real_traffic_v2.py` carries explicit "joblib/pickle is unsafe for untrusted files" warnings (`:402-403,423-424`) — appropriate for a dual-use ML loader of *own* artifacts; no action beyond keeping the warning.
- **Dataset redistribution** (CICIDS2017 2 GB, NSL-KDD 53 MB) in a public repo without documented license clearance (M9) — the most likely "unsafe to publish" item before submission.
- **Deprecated `.pcap` captures** (~33 MB) — confirm they contain no real-host payloads (H6/M).

---

## 11. Prioritized action plan

### Must fix before the thesis defense
1. **Reframe the headline around the split problem (C1).** Promote day / leave-one-CSV-out to co-headline; label the random number "in-distribution / optimistic"; deduplicate before splitting; report duplicate counts. *This is the difference between a defensible and an attackable thesis.*
2. **Disclose `gamma=0` and the contextual-bandit framing (H1)** in the methodology (both languages) and in the defense script; prepare the "why distributional QRDQN still helps at γ=0 / why not just a classifier" answer.
3. **Correct the Phase-2 narrative (H2):** call the data synthetic everywhere, commit/document its generation+labeling pipeline, and stop framing it as captured/real or as external validation.
4. **Produce one leakage-free evaluation of the actual MAIN model (H4)** at full budget (day or LOO), and re-run Check A on MAIN — so the validation ladder validates *MAIN*, not a Feb proxy.
5. **Fix `uv.lock` (H8)** so `uv sync --all-extras` (and CI) actually works.
6. **Reconcile the RF baseline code with its claims (H5)** (balanced+scaled+same-split+artifacts) or soften the text.
7. **Resolve the fixed-partition/SHA-256 story (H7):** either mint+commit the reference manifest or rewrite README/results to stop implying active verification.

### Should fix soon
8. Add balanced-accuracy / MCC / PR-AUC / FPR and lead with operational metrics (M2); add ≥3 seeds or a bootstrap CI (M1).
9. Unify train↔inference preprocessing (H3) and the metric implementations (M5).
10. Add explicit top-of-`main()` seeding (M4); pin CI to Python 3.12 (M10) + cache torch (M11).
11. Migrate raw binaries to LFS or drop archives (H6); untrack `graphify-out/` and regenerate (remove the `fp=-1.5` node) (M6); resolve dataset redistribution (M9).
12. Fix the stale/contradictory docs: `data-structure-...report.md:49` (M14), `results.md:107` (M15), README "fixed partition" (H7), methodology present-tense protocols (M13).
13. Add the missing thesis chapters (Results/Discussion/Limitations/Ethics-synthetic-caveat) tying every number to a RUN_ID; downgrade un-run protocols to future work (M12).
14. Scrub/migrate the email going forward (M8).

### Nice to have
15. Document the missingness-mask semantic (constant=1 on CICIDS2017) (M3); state MAIN hyperparameters are hand-set, not Optuna-derived (M16).
16. Remove dead env branch (L1), tighten the QRDQN→DQN fallback (L2), assert Phase-2 row alignment (L3).
17. Untrack PDFs (L4); fix `requirements-runpod` (L5); add `[tool.ruff]` (L6); comment `graphifyy` (L7); pick a canonical thesis language (L8); sync `GEMINI.md` (L9); archive the deprecated predictor (L10).

---

## 12. Questions for the owner (only the genuinely blocking ones)

1. **Phase-2 data provenance:** how was `pcaps/synthetic_real_traffic.csv` (2M rows) generated, and how are `truth_y`/`source_label` assigned? Is it derived from the CICIDS2017 distribution (→ the "domain shift" claim is circular) or independent? This decides whether the chapters can say anything beyond "synthetic self-check."
2. **Intended claim:** is the defended thesis "high in-distribution detection on CICIDS2017" (random split, defensible-with-caveats) or "generalizes to unseen attack days" (requires a full-budget temporal result as headline)? The required fixes differ.
3. **Was a full-budget (3M-step, `[1024,1024,512]`) day-split or LOO run ever executed but not committed?** If so it is the defensible generalization figure and should be located/committed.
4. **Is `gamma=0` final and deliberate** (one-step contextual bandit)? If yes, the methodology should own it and justify the distributional choice.
5. **Inference clipping:** deliberate inference-only domain-shift mitigation, or an oversight that training should match? (Determines fix direction for H3.)
6. **Which thesis language is canonical** (ES `memoria/` vs EN `report/`), and must both stay in sync? (Determines whether H1/H2/H5 corrections are applied once or twice.)
7. **CICIDS2017/NSL-KDD redistribution:** have you confirmed UNB/CIC terms permit hosting the full datasets in a public repo? If not, switch to a download-script model before submission.

---

*Audit produced read-only. No source, data, config, or thesis files were modified. The two audit sub-tasks that hit a session limit (reproducibility-artifacts sweep, automated cross-check) were reconstructed by the orchestrator from direct file reading and the other sweeps' reproducibility findings.*
