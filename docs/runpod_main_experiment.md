# RunPod Main QRDQN Experiment

This guide prepares one main cloud training run for the thesis model. It is not
a hyperparameter search.

## Machine

- Preferred: RTX 3090. It has 24 GB VRAM and commonly enough system RAM for the
  full CICIDS2017 run.
- Optional: L40S.
- A100 is not necessary for this experiment.

Expected cost is pricing-dependent. A reasonable RTX 3090 budget is likely under
15-20 USD if setup plus training finishes in roughly 8-24 hours.

## Setup

```bash
git clone https://github.com/yeaight7/TFG_CYBER_AI.git
cd TFG_CYBER_AI
python -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Place the CICIDS2017 CSVs here:

```text
datasets/CICIDS2017/*.csv
```

Expected files are the eight official CICIDS2017 CICFlowMeter CSV exports.

## Smoke Check

Run this before the main experiment:

```bash
python src/train_rl_defender.py \
  --preset fast \
  --split-mode random \
  --timesteps 25000 \
  --seed 42
```

For a faster code-path check with the main profile:

```bash
python src/train_rl_defender.py \
  --preset fast \
  --split-mode random \
  --max-rows 10000 \
  --timesteps 1000 \
  --seed 42 \
  --training-profile main-experiment
```

Smoke runs below `learning_starts=50_000` only validate the pipeline and
artifact saving. They do not validate learning quality.

To validate actual learning before the main run, use at least:

```bash
python src/train_rl_defender.py \
  --preset fast \
  --split-mode random \
  --max-rows 250000 \
  --timesteps 150000 \
  --seed 42 \
  --training-profile main-experiment \
  --checkpoint-freq 0
```

## Main Command

```bash
python src/train_rl_defender.py \
  --preset full \
  --split-mode random \
  --timesteps 3000000 \
  --seed 42 \
  --training-profile main-experiment
```

This command uses all available CICIDS2017 rows, canonical schema, random 80/20
split, QRDQN, and the fixed main-experiment hyperparameter profile.

## TensorBoard

```bash
tensorboard --logdir runs/cicids2017
```

The training script writes TensorBoard events while the run is active. This is
the safest way to capture learning curves without adding work inside the
training loop.

After the run, export thesis-ready CSV and PNG curves:

```bash
python scripts/export_tensorboard_scalars.py --run-id <RUN_ID>
```

This reads TensorBoard event files such as `runs/cicids2017/<RUN_ID>_0/` and
writes:

```text
runs/cicids2017/<RUN_ID>/plots/tensorboard_scalars/
```

The exporter also updates `artifact_manifest.json` with the plot directory and
its own export manifest. It can be run while training is still active, but the
final export should be run after training completes.

## Artifacts

The final model is written to:

```text
models/<RUN_ID>.zip
runs/cicids2017/<RUN_ID>/model.zip
```

Run artifacts are written to:

```text
runs/cicids2017/<RUN_ID>/
```

The run directory is intended to be self-contained for frozen-model inference:
it contains the model copy, config, scaler, percentiles, feature names,
environment metadata, metrics, TensorBoard logs, and artifact manifest.

Download these after training:

- `models/<RUN_ID>.zip` or `runs/cicids2017/<RUN_ID>/model.zip`
- `runs/cicids2017/<RUN_ID>/config.json`
- `runs/cicids2017/<RUN_ID>/artifact_manifest.json`
- `runs/cicids2017/<RUN_ID>/environment.json`
- `runs/cicids2017/<RUN_ID>/feature_names.json`
- `runs/cicids2017/<RUN_ID>/metrics.json`
- `runs/cicids2017/<RUN_ID>/scaler.joblib`
- `runs/cicids2017/<RUN_ID>/train_percentiles.npz`
- TensorBoard event files under `runs/cicids2017/<RUN_ID>_0/`
- `runs/cicids2017/<RUN_ID>/plots/tensorboard_scalars/` after running the exporter
- `runs/cicids2017/<RUN_ID>/checkpoints/` only if a checkpoint is needed

Do not commit raw datasets, huge checkpoints, or unnecessary generated binary
artifacts unless intentionally tracked.

Recommended download bundle:

```bash
tar -czf "<RUN_ID>_bundle.tar.gz" \
  "runs/cicids2017/<RUN_ID>" \
  "runs/cicids2017/<RUN_ID>_0" \
  "models/<RUN_ID>.zip"
```

## Post-Training Validation

Use Check A to evaluate the frozen trained model directly on the same split:

```bash
python src/validate_checks.py \
  --model runs/cicids2017/<RUN_ID>/model.zip \
  --checks A \
  --preset full \
  --split-mode random \
  --seed 42
```

Check B and Check C retrain models. Do not run them automatically as part of
this single main cloud experiment. Post-training evaluation of the frozen model
is acceptable; additional training is outside this run.
