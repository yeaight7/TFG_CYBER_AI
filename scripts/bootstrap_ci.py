"""Bootstrap confidence intervals from fresh MAIN persisted predictions.

The maintained CLI accepts only a completed schema-3 ``main-v1`` run and binds
its output to the source manifest and ``predictions.npz`` hashes. Legacy
confusion-count helpers remain importable for historical-result verification,
but the fresh campaign path never treats historical counts as new evidence.

Example:
    python scripts/bootstrap_ci.py --run-dir <FRESH_MAIN_RUN> \
        --output-dir <BOOTSTRAP_JOB_DIR> --n-boot 10000 --boot-seed 12345
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from src.metrics_utils import confusion_to_metrics  # noqa: E402
from src.artifact_integrity import (  # noqa: E402
    resolve_trusted_artifact,
    sha256_file,
    verify_artifact_manifest,
)
from src.run_artifacts import (  # noqa: E402
    ArtifactManifestWriter,
    ArtifactRequirement,
    atomic_write_json,
)

# Metrics to report CIs for (headline + operational). Keys must exist in
# metrics_utils.confusion_to_metrics output.
_REPORT_KEYS = [
    "accuracy",
    "balanced_accuracy",
    "mcc",
    "precision_attack",
    "recall_attack",
    "f1_attack",
    "fpr",
    "fnr",
]

_DEFAULT_RUN = "runs/cicids2017/MAIN_qrdqn_cicids2017_canonical_full_random_20260609_193655"


def _find_main_run(run_arg: str | None) -> Path:
    """Resolve the MAIN run directory (explicit arg, else the single MAIN_* dir)."""
    if run_arg:
        p = (_REPO / run_arg) if not Path(run_arg).is_absolute() else Path(run_arg)
        if not p.is_dir():
            raise FileNotFoundError(f"--run-dir not found: {p}")
        return p
    default = _REPO / _DEFAULT_RUN
    if default.is_dir():
        return default
    matches = sorted((_REPO / "runs" / "cicids2017").glob("MAIN_*"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(
        "Could not auto-resolve the MAIN run dir; pass --run-dir. "
        f"Candidates: {[m.name for m in matches]}"
    )


def recover_confusion_counts(
    metrics: Dict[str, float], split_meta: Dict[str, float]
) -> Tuple[int, int, int, int]:
    """Recover the exact (tn, fp, fn, tp) cell counts from committed artifacts.

    Uses per-class recalls (which are tp/N+ and tn/N-) and the integer class
    totals, so the counts are exact integers. Caller must self-validate.
    """
    n_benign = int(split_meta["test_benign"])
    n_attack = int(split_meta["test_attack"])
    tp = int(round(metrics["recall_attack"] * n_attack))
    fn = n_attack - tp
    tn = int(round(metrics["recall_benign"] * n_benign))
    fp = n_benign - tn
    return tn, fp, fn, tp


def self_validate(
    tn: int, fp: int, fn: int, tp: int,
    published: Dict[str, float], reward_config: Dict[str, float] | None,
    tol: float = 1e-9,
) -> None:
    """Re-derive every published metric from the recovered counts; assert match."""
    derived = confusion_to_metrics(tn, fp, fn, tp, reward_config=reward_config)
    mismatches = []
    for key, pub_val in published.items():
        if key in derived and isinstance(pub_val, (int, float)):
            if abs(float(derived[key]) - float(pub_val)) > tol:
                mismatches.append((key, pub_val, derived[key]))
    if mismatches:
        lines = "\n".join(f"  {k}: published={p!r} derived={d!r}" for k, p, d in mismatches)
        raise AssertionError(
            "Recovered confusion counts do not reproduce the published metrics "
            f"within tol={tol}:\n{lines}"
        )


def bootstrap_metrics(
    tn: int, fp: int, fn: int, tp: int,
    n_boot: int, rng: np.random.Generator,
    reward_config: Dict[str, float] | None,
    stratified: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Percentile bootstrap of confusion-derived metrics with a 95% CI.

    The seed-42 split is stratified, so by default the per-class test totals
    (``N- = tn+fp`` benign, ``N+ = fn+tp`` attack) are held FIXED and the cells
    are resampled within each class (``fp~Binomial(N-,·)``, ``tp~Binomial(N+,·)``)
    — the bootstrap faithful to the stratified sampling design. Pass
    ``stratified=False`` for the unconditional multinomial bootstrap (lets the
    class mix vary; negligibly wider CIs at this n).

    Returns ``{metric_key: {point, boot_mean, boot_std, ci95_low, ci95_high}}``.
    """
    if stratified:
        n_benign = tn + fp
        n_attack = fn + tp
        fp_draws = (rng.binomial(n_benign, fp / n_benign, size=n_boot)
                    if n_benign else np.zeros(n_boot, dtype=np.int64))
        tp_draws = (rng.binomial(n_attack, tp / n_attack, size=n_boot)
                    if n_attack else np.zeros(n_boot, dtype=np.int64))
        # column order = (tn, fp, fn, tp), conditioning on fixed class totals.
        draws = np.column_stack([n_benign - fp_draws, fp_draws, n_attack - tp_draws, tp_draws])
    else:
        n = tn + fp + fn + tp
        probs = np.array([tn, fp, fn, tp], dtype=np.float64) / n
        draws = rng.multinomial(n, probs, size=n_boot)

    # Accumulate each metric across bootstrap replicates.
    samples: Dict[str, np.ndarray] = {k: np.empty(n_boot, dtype=np.float64) for k in _REPORT_KEYS}
    for i in range(n_boot):
        b_tn, b_fp, b_fn, b_tp = (int(v) for v in draws[i])
        m = confusion_to_metrics(b_tn, b_fp, b_fn, b_tp, reward_config=reward_config)
        for k in _REPORT_KEYS:
            samples[k][i] = m[k]

    point = confusion_to_metrics(tn, fp, fn, tp, reward_config=reward_config)
    out: Dict[str, Dict[str, float]] = {}
    for k in _REPORT_KEYS:
        arr = samples[k]
        lo, hi = np.percentile(arr, [2.5, 97.5])
        out[k] = {
            "point": round(float(point[k]), 6),
            "boot_mean": round(float(arr.mean()), 6),
            "boot_std": round(float(arr.std(ddof=1)), 6),
            "ci95_low": round(float(lo), 6),
            "ci95_high": round(float(hi), 6),
        }
    return out


def verify_from_model(
    run_dir: Path,
    expected: Tuple[int, int, int, int],
    *,
    allow_unsafe_artifacts: bool = False,
) -> Dict[str, object]:
    """Re-run the saved MAIN model over the reproduced seed-42 test split and
    assert the regenerated confusion matrix equals `expected`.

    Heavy: loads the CICIDS2017 CSVs, builds the 2.8M-row canonical matrix,
    applies the SAVED scaler (faithful to training), and runs the model.
    Requires the LFS CSVs + the MAIN model.zip + sb3-contrib/torch.
    """
    import joblib  # noqa: PLC0415
    from sklearn.metrics import confusion_matrix  # noqa: PLC0415
    from sb3_contrib import QRDQN  # noqa: PLC0415
    from src.load_cicids2017 import load_cicids2017_split  # noqa: PLC0415

    model_path = _REPO / "models" / f"{run_dir.name}.zip"
    if not model_path.exists():
        run_copy = run_dir / "model.zip"
        if run_copy.exists():
            model_path = run_copy
        else:
            raise FileNotFoundError(f"MAIN model not found at {model_path} or {run_copy}")

    print("[from-model] reproducing seed-42 split (scale=False) ...")
    X_train, y_train, X_test, y_test, _sc, feat, split_meta = load_cicids2017_split(
        split_mode="random", preset="full", seed=42, scale=False, use_canonical=True,
    )
    del X_train, y_train  # only the test split is needed

    model_path = resolve_trusted_artifact(
        run_dir,
        "model",
        model_path,
        repo_root=_REPO,
        allow_unsafe=allow_unsafe_artifacts,
    )
    scaler_path = resolve_trusted_artifact(
        run_dir,
        "scaler",
        run_dir / "scaler.joblib",
        repo_root=_REPO,
        allow_unsafe=allow_unsafe_artifacts,
    )
    scaler = joblib.load(scaler_path)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    print(f"[from-model] loading trusted model artifact and predicting "
          f"({len(y_test):,} rows) ...")
    model = QRDQN.load(str(model_path))
    batch = 8192
    chunks = []
    for s in range(0, len(X_test_scaled), batch):
        actions, _ = model.predict(X_test_scaled[s:s + batch], deterministic=True)
        chunks.append(np.asarray(actions, dtype=np.int64).reshape(-1))
    y_pred = np.concatenate(chunks)

    cm = confusion_matrix(y_test.astype(np.int64), y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    got = (tn, fp, fn, tp)
    matched = got == expected
    print(f"[from-model] regenerated (tn,fp,fn,tp)={got}  expected={expected}  "
          f"{'MATCH' if matched else 'MISMATCH'}")
    if not matched:
        raise AssertionError(
            f"--from-model regenerated confusion {got} != recovered {expected}"
        )

    # Record provenance so the committed result is independently checkable.
    h = __import__("hashlib").sha256()
    with model_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return {
        # "model_file": model_path.name,
        "model_file_sha256": __import__("hashlib").sha256(
            model_path.name.encode("utf-8")
        ).hexdigest(),
        "model_sha256": h.hexdigest(),
        "n_test_predicted": int(len(y_pred)),
        "regenerated_counts": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "matches_recovered": matched,
    }


def run_bootstrap_from_predictions(
    *,
    source_run_dir: Path,
    output_dir: Path,
    n_boot: int = 10_000,
    boot_seed: int = 12_345,
    stratified: bool = True,
    campaign_id: str | None = None,
    logical_run_id: str | None = None,
    attempt: int = 1,
) -> Path:
    """Bootstrap persisted fresh-MAIN y_true/y_pred and bind output to its hash."""
    if n_boot <= 0:
        raise ValueError("n_boot must be greater than zero")
    source_run_dir = Path(source_run_dir)
    output_dir = Path(output_dir)
    verification = verify_artifact_manifest(source_run_dir)
    if verification["schema_version"] != "3.0":
        raise ValueError("Fresh MAIN bootstrap requires a schema-3 source run")
    source_config = json.loads(
        (source_run_dir / "config.json").read_text(encoding="utf-8")
    )
    if source_config.get("profile_id") != "main-v1":
        raise ValueError("Fresh MAIN bootstrap requires profile_id='main-v1'")
    predictions_path = resolve_trusted_artifact(
        source_run_dir,
        "predictions",
        repo_root=_REPO,
    )
    source_predictions_sha256 = sha256_file(predictions_path)
    source_manifest_sha256 = sha256_file(source_run_dir / "artifact_manifest.json")
    predictions = np.load(predictions_path)
    if set(predictions.files) < {"y_true", "y_pred"}:
        raise ValueError("Fresh MAIN predictions.npz must contain y_true and y_pred")
    y_true = np.asarray(predictions["y_true"], dtype=np.int64).reshape(-1)
    y_pred = np.asarray(predictions["y_pred"], dtype=np.int64).reshape(-1)
    if len(y_true) != len(y_pred):
        raise ValueError("Fresh MAIN y_true/y_pred lengths differ")
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    reward_config = source_config.get("reward_config")
    ci = bootstrap_metrics(
        tn,
        fp,
        fn,
        tp,
        n_boot,
        np.random.default_rng(boot_seed),
        reward_config,
        stratified=stratified,
    )
    writer = ArtifactManifestWriter(
        output_dir,
        run_metadata={
            "campaign_id": campaign_id,
            "logical_run_id": logical_run_id or output_dir.name,
            "physical_run_id": output_dir.name,
            "attempt": attempt,
            "split_seed": source_config["split_seed"],
            "model_seed": source_config["model_seed"],
            "source_run_id": source_config["run_id"],
            "source_manifest_sha256": source_manifest_sha256,
        },
        requirements={
            "config": ArtifactRequirement("config.json"),
            "bootstrap_ci": ArtifactRequirement("bootstrap_ci.json"),
        },
    )
    writer.start()
    try:
        config = {
            "job_type": "fresh_main_bootstrap_ci",
            "source_run_id": source_config["run_id"],
            "source_manifest_sha256": source_manifest_sha256,
            "source_predictions_sha256": source_predictions_sha256,
            "bootstrap_seed": boot_seed,
            "n_resamples": n_boot,
            "stratified": stratified,
        }
        result = {
            "source_run_id": source_config["run_id"],
            "source_manifest_sha256": source_manifest_sha256,
            "source_predictions_sha256": source_predictions_sha256,
            "n_test": int(len(y_true)),
            "n_boot": n_boot,
            "boot_seed": boot_seed,
            "bootstrap": "stratified" if stratified else "unstratified",
            "ci_level": 0.95,
            "confusion_counts": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
            "metrics_ci": ci,
        }
        atomic_write_json(output_dir / "config.json", config)
        atomic_write_json(output_dir / "bootstrap_ci.json", result)
        writer.complete()
        return output_dir
    except BaseException as error:
        writer.fail(error)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Bootstrap confidence intervals from fresh MAIN persisted predictions"
    )
    ap.add_argument("--run-dir", required=True, type=Path, help="Fresh schema-3 MAIN run dir")
    ap.add_argument("--output-dir", required=True, type=Path, help="Bootstrap job artifact dir")
    ap.add_argument("--n-boot", type=int, default=10000, help="Bootstrap replicates (default 10000)")
    ap.add_argument("--boot-seed", type=int, default=12345, help="Bootstrap RNG seed (default 12345)")
    ap.add_argument("--unstratified", action="store_true",
                    help="Use the unconditional multinomial bootstrap (default: stratified per-class)")
    ap.add_argument("--campaign-id", default=None)
    ap.add_argument("--logical-run-id", default=None)
    ap.add_argument("--attempt", type=int, default=1)
    return ap.parse_args(argv)


def main() -> None:
    args = parse_args()
    run_bootstrap_from_predictions(
        source_run_dir=args.run_dir,
        output_dir=args.output_dir,
        n_boot=args.n_boot,
        boot_seed=args.boot_seed,
        stratified=not args.unstratified,
        campaign_id=args.campaign_id,
        logical_run_id=args.logical_run_id,
        attempt=args.attempt,
    )


if __name__ == "__main__":
    main()
