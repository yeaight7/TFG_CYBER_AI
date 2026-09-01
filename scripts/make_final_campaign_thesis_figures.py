"""Render thesis-quality final figures from a complete campaign aggregate.

The script is intentionally fail-closed: it validates aggregate checksums and
campaign completeness before importing plotting code or creating an output
directory. It has no example-data or partial-campaign mode.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

if __package__ in {None, ""}:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.campaign_aggregation import (  # noqa: E402
    AGGREGATE_FILES,
    CampaignAggregationError,
    validate_aggregate_directory,
)


BLUE = "#2A78D6"
AQUA = "#1BAF7A"
RED = "#E34948"
INK = "#0B0B0B"
MUTED = "#52514E"
GRID = "#E1E0D9"
OUTPUT_STEMS = (
    "final_main_confusion_matrix",
    "final_main_bootstrap_intervals",
    "final_main_duplicate_analysis",
    "final_day_generalisation",
    "final_size_ladder",
    "final_seed_sensitivity",
    "final_targeted_holdouts",
    "final_qrdqn_vs_rf",
    "final_phase2_diagnostics",
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render PDF/SVG/PNG thesis figures only from a checksum-validated, "
            "complete final-campaign aggregate."
        )
    )
    parser.add_argument("--aggregate-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CampaignAggregationError(f"Expected JSON object: {path.name}")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CampaignAggregationError(f"Missing numeric value for {label}")
    return float(value)


def _csv_number(value: Any, *, label: str) -> float:
    if value in (None, ""):
        raise CampaignAggregationError(f"Missing numeric value for {label}")
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise CampaignAggregationError(f"Invalid numeric value for {label}") from error


def _rows(document: Mapping[str, Any], *, label: str) -> list[dict[str, Any]]:
    value = document.get("rows")
    if not isinstance(value, list) or not value or not all(isinstance(row, dict) for row in value):
        raise CampaignAggregationError(f"Missing aggregate rows for {label}")
    return value


def _metric(row: Mapping[str, Any], name: str) -> float:
    metrics = row.get("metrics")
    if not isinstance(metrics, Mapping):
        raise CampaignAggregationError(f"Missing metrics for {row.get('logical_run_id')}")
    return _number(metrics.get(name), label=f"{row.get('logical_run_id')}.{name}")


def _style(plt: Any) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "grid.alpha": 1.0,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
        }
    )


def _despine(ax: Any) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save(fig: Any, directory: Path, stem: str) -> list[str]:
    produced: list[str] = []
    for suffix in ("pdf", "svg", "png"):
        path = directory / f"{stem}.{suffix}"
        options: dict[str, Any] = {"dpi": 240}
        if suffix == "pdf":
            options["metadata"] = {
                "Title": stem,
                "Author": "TFG_CYBER_AI",
                "Creator": "make_final_campaign_thesis_figures.py",
                "CreationDate": None,
                "ModDate": None,
            }
        elif suffix == "svg":
            options["metadata"] = {"Title": stem, "Date": None}
        elif suffix == "png":
            options["metadata"] = {"Software": "make_final_campaign_thesis_figures.py"}
        fig.savefig(path, format=suffix, **options)
        produced.append(path.name)
    return produced


def _grouped_bars(
    ax: Any,
    labels: Sequence[str],
    left: Sequence[float],
    right: Sequence[float],
    *,
    left_label: str = "QRDQN",
    right_label: str = "Random Forest",
) -> None:
    import numpy as np

    x = np.arange(len(labels))
    width = 0.36
    ax.bar(x - width / 2, left, width, color=BLUE, label=left_label)
    ax.bar(x + width / 2, right, width, color=AQUA, label=right_label)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.03)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    _despine(ax)


def _fig_main_cm(plt: Any, data: Mapping[str, Any]) -> Any:
    import numpy as np

    row = _rows(data["main"], label="main")[0]
    matrix = np.array(
        [
            [_metric(row, "tn"), _metric(row, "fp")],
            [_metric(row, "fn"), _metric(row, "tp")],
        ]
    )
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    ax.imshow(matrix, cmap="Blues")
    total = matrix.sum()
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{int(matrix[i, j]):,}\n{matrix[i, j] / total:.2%}",
                ha="center",
                va="center",
                color="white" if matrix[i, j] > matrix.max() * 0.45 else INK,
                fontweight="bold",
            )
    ax.set_xticks([0, 1], ["PERMIT", "BLOCK"])
    ax.set_yticks([0, 1], ["BENIGN", "ATTACK"])
    ax.set_xlabel("Decisión")
    ax.set_ylabel("Clase real")
    ax.set_title("MAIN final · matriz de confusión")
    ax.grid(False)
    return fig


def _fig_bootstrap(plt: Any, data: Mapping[str, Any]) -> Any:
    result = data["bootstrap"].get("result")
    ci = result.get("metrics_ci") if isinstance(result, Mapping) else None
    wanted = [
        ("accuracy", "Exactitud"),
        ("balanced_accuracy", "Exactitud bal."),
        ("mcc", "MCC"),
        ("recall_attack", "Recall ataque"),
        ("f1_attack", "F1 ataque"),
        ("fpr", "FPR"),
        ("fnr", "FNR"),
    ]
    available = [(key, label, ci.get(key)) for key, label in wanted if isinstance(ci, Mapping) and isinstance(ci.get(key), Mapping)]
    if not available:
        raise CampaignAggregationError("Fresh MAIN bootstrap has no plottable intervals")
    labels, points, low, high = [], [], [], [], []
    for key, label, record in available:
        labels.append(label)
        points.append(_number(record.get("point"), label=f"bootstrap.{key}.point"))
        low.append(_number(record.get("ci95_low"), label=f"bootstrap.{key}.ci95_low"))
        high.append(_number(record.get("ci95_high"), label=f"bootstrap.{key}.ci95_high"))
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    y = list(range(len(labels)))
    ax.errorbar(points, y, xerr=[[p - lo for p, lo in zip(points, low)], [hi - p for p, hi in zip(points, high)]], fmt="o", color=BLUE, ecolor=BLUE, capsize=3)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Valor e IC bootstrap al 95 %")
    ax.set_title("MAIN final · precisión sobre el test fijo")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    _despine(ax)
    return fig


def _fig_duplicates(plt: Any, data: Mapping[str, Any]) -> Any:
    result = data["duplicates"].get("result")
    if not isinstance(result, Mapping):
        raise CampaignAggregationError("Fresh MAIN duplicate result is missing")
    leakage = result.get("cross_split_leakage_feature_only")
    if not isinstance(leakage, Mapping):
        raise CampaignAggregationError("Fresh MAIN cross-split duplicate result is missing")
    labels = ["Test ya presente\nen train"]
    values = [_number(leakage.get("pct_of_test"), label="duplicates.pct_of_test")]
    for key, label in (
        ("duplicate_rows_full_pct", "Duplicados\nen total"),
        ("pct_of_test_attack", "Ataques test\nya en train"),
        ("pct_of_test_benign", "Benignos test\nya en train"),
    ):
        if isinstance(result.get(key), (int, float)) and not isinstance(result.get(key), bool):
            labels.append(label)
            values.append(float(result[key]))
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    bars = ax.bar(labels, values, color=[BLUE, AQUA, RED, "#898781"][: len(values)])
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f} %", ha="center", va="bottom")
    ax.set_ylabel("Porcentaje de filas")
    ax.set_title("MAIN final · duplicados exactos")
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    _despine(ax)
    return fig


def _fig_day(plt: Any, data: Mapping[str, Any]) -> Any:
    rows = _rows(data["day"], label="day split")
    by_model = {row.get("model_family"): row for row in rows}
    qrdqn, rf = by_model.get("qrdqn"), by_model.get("random_forest")
    if not isinstance(qrdqn, Mapping) or not isinstance(rf, Mapping):
        raise CampaignAggregationError("Day split lacks matched QRDQN/RF rows")
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    _grouped_bars(ax, ["Recall ataque", "F1 ataque", "Exactitud bal."], [_metric(qrdqn, "recall_attack"), _metric(qrdqn, "f1_attack"), _metric(qrdqn, "balanced_accuracy")], [_metric(rf, "recall_attack"), _metric(rf, "f1_attack"), _metric(rf, "balanced_accuracy")])
    ax.set_title("Partición completa por días")
    ax.legend(loc="lower right")
    return fig


def _fig_ladder(plt: Any, data: Mapping[str, Any]) -> Any:
    rows = _rows(data["ladder"], label="size ladder")
    x = [int(row.get("train_rows") or row.get("train_max_rows")) for row in rows]
    fig, ax = plt.subplots(figsize=(6.1, 3.5))
    ax.plot(x, [_metric(row, "f1_attack") for row in rows], marker="o", color=BLUE, label="F1 ataque")
    ax.plot(x, [_metric(row, "recall_attack") for row in rows], marker="s", color=AQUA, label="Recall ataque")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Filas de entrenamiento (escala logarítmica)")
    ax.set_ylabel("Métrica")
    ax.set_title("QRDQN · escalera de tamaño y presupuesto proporcional")
    ax.legend(loc="lower right")
    _despine(ax)
    return fig


def _fig_seeds(plt: Any, data: Mapping[str, Any]) -> Any:
    rows = _rows(data["seeds"], label="seed sensitivity")
    seeds = [int(row["model_seed"]) for row in rows]
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    ax.plot(seeds, [_metric(row, "f1_attack") for row in rows], marker="o", color=BLUE, label="F1 ataque")
    ax.plot(seeds, [_metric(row, "recall_attack") for row in rows], marker="s", color=AQUA, label="Recall ataque")
    ax.set_xticks(seeds)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Semilla del modelo (split seed = 42)")
    ax.set_ylabel("Métrica")
    ax.set_title("QRDQN · sensibilidad a semilla en 1M de filas")
    ax.legend(loc="lower right")
    _despine(ax)
    return fig


def _fig_holdouts(plt: Any, data: Mapping[str, Any]) -> Any:
    rows = [row for row in data["comparisons"] if str(row.get("comparison_id", "")).startswith("holdout_")]
    if len(rows) != 4:
        raise CampaignAggregationError("Expected four matched targeted holdouts")
    names = [str(row["comparison_id"]).removeprefix("holdout_").replace("infilteration", "infiltration").title() for row in rows]
    fig, ax = plt.subplots(figsize=(6.3, 3.5))
    _grouped_bars(ax, names, [_csv_number(row.get("qrdqn_f1_attack"), label="holdout qrdqn f1") for row in rows], [_csv_number(row.get("rf_f1_attack"), label="holdout rf f1") for row in rows])
    ax.set_title("Cuatro escenarios retenidos · F1 de ataque")
    ax.legend(loc="lower right")
    return fig


def _fig_comparison(plt: Any, data: Mapping[str, Any]) -> Any:
    rows = data["comparisons"]
    labels = [str(row["comparison_id"]).replace("holdout_", "").replace("random_", "random ").replace("day_full", "día") for row in rows]
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 5.7), sharex=True)
    for ax, metric, title in ((axes[0], "recall_attack", "Recall de ataque"), (axes[1], "f1_attack", "F1 de ataque")):
        _grouped_bars(ax, labels, [_csv_number(row.get(f"qrdqn_{metric}"), label=f"qrdqn {metric}") for row in rows], [_csv_number(row.get(f"rf_{metric}"), label=f"rf {metric}") for row in rows])
        ax.set_title(title, loc="left")
    axes[0].legend(loc="lower right")
    axes[1].tick_params(axis="x", rotation=25)
    fig.suptitle("Comparaciones emparejadas QRDQN--Random Forest", y=1.01)
    fig.tight_layout()
    return fig


def _fig_phase2(plt: Any, data: Mapping[str, Any]) -> Any:
    result = data["phase2"].get("result")
    diagnostics = data["phase2"].get("diagnostics")
    if not isinstance(result, Mapping) or not isinstance(diagnostics, Mapping):
        raise CampaignAggregationError("Fresh Phase 2 result or diagnostics are missing")
    candidates = [("accuracy", "Exactitud"), ("recall_attack", "Recall ataque"), ("f1_attack", "F1 ataque"), ("block_rate", "Tasa de bloqueo")]
    metrics = [(key, label, float(result[key])) for key, label in candidates if isinstance(result.get(key), (int, float)) and not isinstance(result.get(key), bool)]
    if not metrics:
        raise CampaignAggregationError("Fresh Phase 2 has no plottable metrics")
    fig, ax = plt.subplots(figsize=(5.8, 3.5))
    bars = ax.bar([label for _, label, _ in metrics], [value for _, _, value in metrics], color=[BLUE, AQUA, BLUE, RED][: len(metrics)])
    for bar, (_, _, value) in zip(bars, metrics):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom")
    ax.set_ylim(0, 1.08)
    ax.set_title("Fase 2 fresca · inferencia offline")
    ax.set_ylabel("Valor")
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    note = []
    for key, label in (("z_abs_max", "máx |z|"), ("z_abs_mean", "media |z|")):
        if isinstance(diagnostics.get(key), (int, float)):
            note.append(f"{label}: {float(diagnostics[key]):.3f}")
    if note:
        ax.text(0.99, 0.02, " · ".join(note), transform=ax.transAxes, ha="right", va="bottom", color=MUTED, fontsize=8)
    _despine(ax)
    return fig


def _load_data(aggregate_dir: Path) -> dict[str, Any]:
    return {
        "main": _read_json(aggregate_dir / "main.json"),
        "bootstrap": _read_json(aggregate_dir / "main_bootstrap_ci.json"),
        "duplicates": _read_json(aggregate_dir / "main_duplicate_analysis.json"),
        "day": _read_json(aggregate_dir / "day_split.json"),
        "ladder": _read_json(aggregate_dir / "size_ladder.json"),
        "seeds": _read_json(aggregate_dir / "seed_sensitivity.json"),
        "phase2": _read_json(aggregate_dir / "phase2_fresh_main.json"),
        "comparisons": _read_csv(aggregate_dir / "qrdqn_vs_rf.csv"),
    }


def _check_destination(output_dir: Path) -> None:
    resolved = output_dir.resolve()
    repo = Path(__file__).resolve().parent.parent
    forbidden = ((repo / "memoria").resolve(), (repo / "report").resolve())
    if any(resolved == root or root in resolved.parents for root in forbidden):
        raise CampaignAggregationError("Render to a staging directory outside memoria/ and report/")
    if resolved.exists():
        raise CampaignAggregationError("Output directory must not already exist")


def render(aggregate_dir: Path, output_dir: Path) -> dict[str, Any]:
    aggregate_dir = aggregate_dir.resolve()
    output_dir = output_dir.resolve()
    summary = validate_aggregate_directory(aggregate_dir)
    _check_destination(output_dir)
    data = _load_data(aggregate_dir)

    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style(plt)
    builders: Sequence[tuple[str, Callable[[Any, Mapping[str, Any]], Any]]] = (
        (OUTPUT_STEMS[0], _fig_main_cm),
        (OUTPUT_STEMS[1], _fig_bootstrap),
        (OUTPUT_STEMS[2], _fig_duplicates),
        (OUTPUT_STEMS[3], _fig_day),
        (OUTPUT_STEMS[4], _fig_ladder),
        (OUTPUT_STEMS[5], _fig_seeds),
        (OUTPUT_STEMS[6], _fig_holdouts),
        (OUTPUT_STEMS[7], _fig_comparison),
        (OUTPUT_STEMS[8], _fig_phase2),
    )
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        files: list[str] = []
        for stem, builder in builders:
            figure = builder(plt, data)
            files.extend(_save(figure, temporary, stem))
            plt.close(figure)
        manifest = {
            "schema_version": 1,
            "campaign_id": summary.get("campaign_id"),
            "campaign_complete": True,
            "aggregate_dir": str(aggregate_dir),
            "aggregate_inputs": {
                filename: {"sha256": _sha256(aggregate_dir / filename), "size_bytes": (aggregate_dir / filename).stat().st_size}
                for filename in AGGREGATE_FILES
            },
            "outputs": {
                filename: {"sha256": _sha256(temporary / filename), "size_bytes": (temporary / filename).stat().st_size}
                for filename in files
            },
            "notes": "All geometry is rendered from the validated aggregate; no example-data mode exists.",
        }
        (temporary / "figure_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {"output_dir": str(output_dir), "figures": list(OUTPUT_STEMS), "manifest": "figure_manifest.json"}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = render(args.aggregate_dir, args.output_dir)
    except (CampaignAggregationError, OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
