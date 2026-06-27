# Reproducibility Notes

## Main QRDQN RunPod Environment

The current source of truth for the successful main QRDQN training environment is:

```text
runs/cicids2017/MAIN_qrdqn_cicids2017_canonical_full_random_20260609_193655/environment.json
```

Recorded runtime:

| Component | Version / value |
|-----------|-----------------|
| Python | 3.12.3 |
| Platform | Linux x86_64 |
| Device | CUDA, NVIDIA GeForce RTX 3090 |
| torch | 2.12.0+cu130 |
| CUDA reported by torch | 13.0 |
| numpy | 2.4.6 |
| pandas | 3.0.3 |
| scikit-learn | 1.9.0 |
| gymnasium | 1.2.3 |
| stable-baselines3 | 2.8.0 |
| sb3-contrib | 2.8.0 |
| joblib | 1.5.3 |

## Dependency Files

- `requirements-runpod-cu130.txt` is the direct RunPod/GPU reproduction file for the main QRDQN stack.
- `requirements.txt` is the generic local/dev install file. It pins the same core ML stack but uses `torch==2.12.0` without a CUDA local-version suffix so pip can select the platform-appropriate wheel.
- `pyproject.toml` pins the same core project dependencies. For uv, `tool.uv.sources` sends Linux torch resolution to the PyTorch CUDA 13.0 index and non-Linux torch resolution to the CPU index.

RunPod setup:

```bash
python -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements-runpod-cu130.txt
```

uv setup:

```bash
uv sync
```

On Linux, uv resolves torch through:

```text
https://download.pytorch.org/whl/cu130
```

If using `uv pip install` directly against `requirements-runpod-cu130.txt`,
include uv's multi-index strategy flag:

```bash
uv pip install -r requirements-runpod-cu130.txt --index-strategy unsafe-best-match
```

The project-mode `uv sync` path does not need that flag because `pyproject.toml`
uses per-package `tool.uv.sources` for torch.

## PyTorch Advisory Handling

PyTorch is an intentional dependency. The project imports `torch` directly and QRDQN/SB3 require it.

Repository scan status:

- No direct `torch.jit` usage.
- No direct `torch.jit.script` usage.
- No direct TorchScript save/load path.
- No direct `torch.load` usage.
- Model loading uses `QRDQN.load` / `DQN.load`, and scaler loading uses `joblib.load`; these are trusted local artifact paths only.

This does not claim the upstream PyTorch advisory is fixed. The current handling is: keep the working ML stack pinned, avoid untrusted model/checkpoint/scaler loading, and treat any unresolved upstream PyTorch advisory as residual upstream risk until a compatible patched PyTorch build exists and is validated against the training stack.

## Preprocessing: training vs Phase-2 inference

Training and the internal CICIDS2017 test evaluation use **StandardScaler only, with no clipping**: `src/train_rl_defender.py` computes the `p0.5 / p99.5` train percentiles and persists them, but the model is fit and evaluated on un-clipped, standardized features.

Phase-2 offline inference (`scripts/predict_real_traffic_v2.py`) optionally adds **percentile clipping** (to the persisted train percentiles) and **z-score clipping** (`--clip-z`, e.g. `10.0`) around that same persisted scaler. These are a **deliberate inference-time domain-shift mitigation** for out-of-distribution lab traffic (extreme `|z|` values), not part of the training preprocessing.

Implication: a Phase-2 run that uses `--percentiles` / `--clip-z` applies a transform the model did not see at train time, so its metrics are **not byte-for-byte comparable** with the internal CICIDS2017 test metrics. For a strictly comparable Phase-2 run, omit `--percentiles` and `--clip-z` (matching the un-clipped training preprocessing). The MAIN Phase-2 artifact (`runs/phase2/P2v2_pred_20260610_161231_MAIN/`) used `--clip-z 10.0`, recorded in its `config.json`. In practice the operator observed training and lab-inference accuracy to be close (~0.98–0.99), so the clipping was retained as-is; this paragraph documents the asymmetry rather than hiding it.
