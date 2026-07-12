#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/datasets/CICIDS2017}"
CACHE_ROOT="${CACHE_ROOT:-${REPO_ROOT}/.cache/cicids2017}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${REPO_ROOT}/runs/cicids2017}"
RUN_ID="${RUN_ID:-qrdqn_main_random_full_s42_m42_$(date -u +%Y%m%d_%H%M%S)}"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not found; continuing without GPU info."
fi

if [[ -f "venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "venv/bin/activate"
elif [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
else
  echo "No venv found. Activate your Python environment before running if needed."
fi

python src/train_rl_defender.py \
  --split-mode random \
  --split-seed 42 \
  --model-seed 42 \
  --profile main-v1 \
  --timesteps 3000000 \
  --dataset-root "${DATASET_ROOT}" \
  --cache-root "${CACHE_ROOT}" \
  --cache-policy require \
  --artifact-root "${ARTIFACT_ROOT}" \
  --run-id "${RUN_ID}" \
  --checkpoint-freq 500000 \
  --checkpoint-keep 2 \
  --monitor-interval 30
