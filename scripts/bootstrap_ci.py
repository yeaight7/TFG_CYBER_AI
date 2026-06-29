"""
bootstrap_ci.py — Task A4 (no-retrain).

Puts a confidence interval (CI = *confidence interval*, NOT CI/CD) around the
existing MAIN test metrics by bootstrap-resampling the fixed seed-42 test set.
This answers the "single seed — is 0.9938 just a lucky number?" question
WITHOUT retraining: it characterises the sampling precision of the published
point estimates on the test set the headline model was evaluated on.

Why this needs no GPU / LFS by default
---------------------------------------
Every metric the project reports (accuracy, recall_attack, precision_attack,
f1, FPR, FNR, balanced-accuracy, MCC, ...) is a deterministic function of the
four confusion-matrix cell counts ``(tn, fp, fn, tp)``. The seed-42 split is
*stratified* (``src/load_cicids2017.py``), so the per-class test totals
(``N- = tn+fp`` benign, ``N+ = fn+tp`` attack) are fixed by design. The
bootstrap faithful to that design resamples WITHIN each class:

    fp ~ Binomial(N-, fp/N-),  tn = N- - fp     (benign rows)
    tp ~ Binomial(N+, tp/N+),  fn = N+ - tp     (attack rows)

i.e. it conditions on the fixed class totals (the row-level with-replacement
bootstrap, restricted to within-class resampling, is identically distributed to
these two binomials). Because every metric is a function of the four cells, the
full bootstrap distribution follows from these per-class draws — no model, no
data, no per-row resampling needed. (``--unstratified`` uses the unconditional
``Multinomial(n, cells/n)`` bootstrap instead; it lets the class mix vary and
gives negligibly wider CIs at this n.)

The exact cell counts are recovered from the committed MAIN artifacts: the
**full-precision float64** per-class recalls in ``metrics.json`` (e.g.
``recall_attack = 0.995355468084534``) times the integer class totals in
``config.json`` land exactly on integers. (Recovery is safe ONLY at full
precision — the rounded values shown in result tables would be off by one, e.g.
``round(0.99536 * 111529) = 111012`` vs the true ``111011``.) The recovered
counts are then *self-validated*: every published metric is re-derived from the
recovered integers and asserted to match ``metrics.json`` to 1e-9 — so if that
passes, the counts are provably the ones the headline used.

``--from-model`` (optional, needs the CICIDS2017 LFS CSVs + the MAIN model)
re-runs the actual saved model over the reproduced seed-42 test split and
asserts the regenerated confusion matrix equals the recovered counts —
closing the loop end-to-end.

Run:
    python scripts/bootstrap_ci.py                 # instant; counts mode
    python scripts/bootstrap_ci.py --from-model     # end-to-end re-prediction
    python scripts/bootstrap_ci.py --n-boot 10000 --boot-seed 12345
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from metrics_utils import confusion_to_metrics  # noqa: E402
from artifact_integrity import resolve_trusted_artifact  # noqa: E402

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
    from load_cicids2017 import load_cicids2017_split  # noqa: PLC0415

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

    print(f"[from-model] loading model {model_path.name} and predicting "
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
        "model_file": model_path.name,
        "model_sha256": h.hexdigest(),
        "n_test_predicted": int(len(y_pred)),
        "regenerated_counts": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "matches_recovered": matched,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Bootstrap CI for the MAIN test metrics (no retrain)")
    ap.add_argument("--run-dir", default=None, help="MAIN run dir (default: auto-resolve)")
    ap.add_argument("--n-boot", type=int, default=10000, help="Bootstrap replicates (default 10000)")
    ap.add_argument("--boot-seed", type=int, default=12345, help="Bootstrap RNG seed (default 12345)")
    ap.add_argument("--from-model", action="store_true",
                    help="Also re-run the saved model over the reproduced split and verify counts")
    ap.add_argument("--unstratified", action="store_true",
                    help="Use the unconditional multinomial bootstrap (default: stratified per-class)")
    ap.add_argument("--out", default="runs/validation/bootstrap_ci_seed42.json", help="Output JSON path")
    ap.add_argument("--allow-unsafe-artifacts", action="store_true",
                    help="Allow --from-model to load artifacts without manifest hash verification")
    args = ap.parse_args()

    run_dir = _find_main_run(args.run_dir)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    split_meta = config["split_metadata"]
    reward_config = config.get("reward_config")

    print("=" * 72)
    print("  A4 — Bootstrap CI for the MAIN test metrics (no retrain)")
    print("=" * 72)
    print(f"run: {run_dir.name}")

    # 1) recover + self-validate the exact confusion counts
    tn, fp, fn, tp = recover_confusion_counts(metrics, split_meta)
    self_validate(tn, fp, fn, tp, metrics, reward_config)
    n_test = tn + fp + fn + tp
    print("\nRecovered confusion (self-validated vs metrics.json, tol=1e-9):")
    print(f"  tn={tn:,}  fp={fp:,}  fn={fn:,}  tp={tp:,}   n_test={n_test:,}")
    assert n_test == int(split_meta["n_test"]), "recovered n_test != config split_metadata n_test"

    # 2) bootstrap
    stratified = not args.unstratified
    rng = np.random.default_rng(args.boot_seed)
    kind = "stratified per-class" if stratified else "unconditional multinomial"
    print(f"\nBootstrapping {args.n_boot:,} {kind} resamples (seed={args.boot_seed}) ...")
    ci = bootstrap_metrics(tn, fp, fn, tp, args.n_boot, rng, reward_config, stratified=stratified)

    print("\n--- 95% bootstrap CI (point [ci_low, ci_high]) ---")
    for k in _REPORT_KEYS:
        c = ci[k]
        print(f"  {k:<18} {c['point']:.6f}  [{c['ci95_low']:.6f}, {c['ci95_high']:.6f}]")

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "task": "A4 - bootstrap CI of MAIN test metrics (no retrain)",
        "run_id": config.get("run_id", run_dir.name),
        "bootstrap": "stratified" if stratified else "unstratified",
        "method": (
            ("Stratified " if stratified else "Unconditional multinomial ")
            + "nonparametric percentile bootstrap of the fixed seed-42 test set. "
            + ("Per-class resampling conditioning on the fixed stratified class totals "
               "(fp~Binomial(N-,.), tp~Binomial(N+,.)). " if stratified else
               "Multinomial(n, cells/n) resampling. ")
            + "Every reported metric is a function of the confusion cells (tn,fp,fn,tp)."
        ),
        "n_test": n_test,
        "n_boot": args.n_boot,
        "boot_seed": args.boot_seed,
        "ci_level": 0.95,
        "confusion_counts": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "counts_self_validated": True,
        "metrics_ci": ci,
    }

    if args.from_model:
        summary["model_verification"] = verify_from_model(
            run_dir,
            (tn, fp, fn, tp),
            allow_unsafe_artifacts=args.allow_unsafe_artifacts,
        )

    out_path = (_REPO / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[output] {out_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
