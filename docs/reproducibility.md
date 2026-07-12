# Reproducibility Notes

## GPU experimental environment

The final campaign uses a provider-neutral Linux GPU environment. The maintained pinned direct dependency set is `requirements-gpu-cu130.txt`; `pyproject.toml` and `uv.lock` define the development/test environment. Host identity, remote access, account details, and mount paths are operational inputs, not scientific configuration.

Setup, preflight, cache, campaign, snapshot, and aggregation commands live in [gpu_experimental_environment.md](gpu_experimental_environment.md). Actual CPU, RAM, GPU, driver, CUDA, cuDNN, storage, dataset, cache, and snapshot readiness remain unverified until a successful preflight report is produced on the final host.

## Dependency Files

- `requirements-gpu-cu130.txt` is the direct Linux GPU environment file for the QRDQN campaign stack.
- `requirements.txt` is the generic local/dev install file. It pins the same core ML stack but uses `torch==2.12.1` without a CUDA local-version suffix so pip can select the platform-appropriate wheel.
- `pyproject.toml` pins the same core project dependencies. For uv, `tool.uv.sources` sends Linux torch resolution to the PyTorch CUDA 13.0 index and non-Linux torch resolution to the CPU index.

GPU host setup:

```bash
python -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements-gpu-cu130.txt
```

uv setup:

```bash
uv sync
```

On Linux, uv resolves torch through:

```text
https://download.pytorch.org/whl/cu130
```

If using `uv pip install` directly against `requirements-gpu-cu130.txt`,
include uv's multi-index strategy flag:

```bash
uv pip install -r requirements-gpu-cu130.txt --index-strategy unsafe-best-match
```

The project-mode `uv sync` path does not need that flag because `pyproject.toml`
uses per-package `tool.uv.sources` for torch.

## Historical MAIN environment and compatibility filename

The committed historical MAIN environment remains recorded in:

```text
runs/cicids2017/MAIN_qrdqn_cicids2017_canonical_full_random_20260609_193655/environment.json
```

That artifact records the original RunPod execution and its measured hardware/software metadata. It is evidence for the historical MAIN only; it does not verify or constrain the future campaign host. `requirements-runpod-cu130.txt` is retained as a compatibility filename and includes the provider-neutral `requirements-gpu-cu130.txt` file.

## PyTorch Advisory Handling

PyTorch is an intentional dependency. The project imports `torch` directly and QRDQN/SB3 require it.

Repository scan status:

- No direct `torch.jit` usage.
- No direct `torch.jit.script` usage.
- No direct TorchScript save/load path.
- No direct `torch.load` usage.
- Model loading uses `QRDQN.load` / `DQN.load`, and scaler loading uses `joblib.load`; default CLI paths must resolve through `artifact_manifest.json` and pass SHA-256 verification before deserialization. Ad-hoc local artifacts require the explicit `--allow-unsafe-artifacts` override.

## Preprocessing: training vs Phase-2 inference

Training and the internal CICIDS2017 test evaluation use **StandardScaler only, with no clipping**: `src/train_rl_defender.py` computes the `p0.5 / p99.5` train percentiles and persists them, but the model is fit and evaluated on un-clipped, standardized features.

Phase-2 offline inference (`scripts/predict_real_traffic_v2.py`) optionally adds **percentile clipping** (to the persisted train percentiles) and **z-score clipping** (`--clip-z`, e.g. `10.0`) around that same persisted scaler. These are a **deliberate inference-time domain-shift mitigation** for out-of-distribution lab traffic (extreme `|z|` values), not part of the training preprocessing.

Implication: a Phase-2 run that uses `--percentiles` / `--clip-z` applies a transform the model did not see at train time, so its metrics are **not byte-for-byte comparable** with the internal CICIDS2017 test metrics. For a strictly comparable Phase-2 run, omit `--percentiles` and `--clip-z` (matching the un-clipped training preprocessing). The MAIN Phase-2 artifact (`runs/phase2/P2v2_pred_20260610_161231_MAIN/`) used `--clip-z 10.0`, recorded in its `config.json`. In practice the operator observed training and lab-inference accuracy to be close (~0.98–0.99), so the clipping was retained as-is; this paragraph documents the asymmetry rather than hiding it.

## Dataset attribution, terms, and provenance

- **CICIDS2017** — Canadian Institute for Cybersecurity (CIC), University of New Brunswick. Official page: <https://www.unb.ca/cic/datasets/ids-2017.html>. The dataset is provided for **research use** and requires **citation/attribution**; redistribution is not explicitly granted. The curated CSVs in `datasets/CICIDS2017/` (git LFS) are a reproducibility convenience (this repo's derivative after pre-ingestion column removal); per-file SHA-256 hashes are in `README.md` (§ Provenance and integrity). Prefer fetching from the official source where possible, and cite CIC/UNB in any derived work.
- **NSL-KDD** (legacy) was **removed** from the repository on 2026-06-27 (decision D-8): `datasets/nsl_kdd/` and `models/rf_nslkdd.joblib` are no longer tracked and are now gitignored. The adapter `src/load_nsl_kdd.py` is kept for historical reference only (Phase-1 benchmark); it is not part of the current CICIDS2017 + Phase-2 model path.

## Git history note (institutional email)

Commits **before 2026-06-27** were authored with an institutional email (`…@al.uloyola.es`). Going forward, commits use the author's GitHub `noreply` address (decision D-6). Per D-6 the existing history is **accepted and not rewritten** (no `git filter-repo` / force-push): the historical address remains in past commits by design, documented here rather than scrubbed.

## Continuous integration: what CI does and does not verify

CI (`.github/workflows/ci.yml`) runs on Python **3.12** (matching the MAIN training environment, `environment.json` 3.12.11), installs the locked dependencies with `uv sync --all-extras`, and runs the unit tests, the canonical-schema dimension check, and Ruff. The uv download/build store is cached (`enable-cache: true`) so the large `cu130` torch wheel (~2.5 GB) is fetched only once.

**What CI cannot verify:** the byte-identical **SHA-256 of the fixed seed-42 test partition** (`runs/cicids2017/test_partition_reference_seed42.json`) can only be checked against the real CICIDS2017 CSVs, which live in **git LFS and are not pulled in CI** (large, and gated by dataset terms). So the reproducibility hash is a **local** check.

**What CI does verify (the split/hash logic, on synthetic data):** `tests/test_load_cicids2017.py` exercises split determinism and invariants without the real dataset — `test_nested_prefix_indices_deterministic`, `test_train_max_rows_keeps_test_set_identical`, `test_scale_true_refits_on_subsample`, and `test_sha256_of_array_stable`. These run in CI, so a regression in the partition/scaler/hashing logic would be caught even though the real-data hash is not recomputed.

**To verify the real-data hash locally:** `git lfs pull`, then run

```bash
python scripts/verify_fixed_test_split.py            # counts + SHA-256 + scaler match
python scripts/verify_fixed_test_split.py --skip-count-check   # hash/scaler only (still needs the CSVs)
```

## Thesis language: canonical source (ES) vs translation (EN)

`memoria/` (Spanish) is the **canonical** thesis — it is the most complete and current version (the defended document) and carries all audit-remediation corrections. `report/` (English) is a **secondary translation** that is re-synced *after* the Spanish source is corrected. When the two disagree, `memoria/` is authoritative.

As of the Phase-4 remediation, `memoria/` adds the Resultados, Discusión, Limitaciones and Consideraciones éticas chapters and the gamma=0 / hyperparameter-provenance / mask-constant-on-CICIDS / closed-lab-Phase-2 corrections. The English `report/` re-sync of these items is pending.
