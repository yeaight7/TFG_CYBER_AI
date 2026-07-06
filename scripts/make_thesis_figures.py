#!/usr/bin/env python
"""Generate the thesis data figures (memoria/figuras/) from committed run artifacts.

Every numeric value plotted by this script is READ from a committed artifact
(JSON / exported TensorBoard CSV / dataset CSV) -- no hand-typed metrics.
Each figure writes a PDF (included by LaTeX) and a PNG (preview), plus an entry
in memoria/figuras/figures_manifest.json recording its sources and the exact
values plotted (gates G4/G7 of docs/thesis/THESIS_IMPROVEMENT_PLAN.md).

Usage:
    uv run python scripts/make_thesis_figures.py               # all figures
    uv run python scripts/make_thesis_figures.py --only f9_cm_main f12_duplicados
    uv run python scripts/make_thesis_figures.py --skip-eda    # skip the dataset scan (F6)
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Reproducible PDF binaries across re-runs (matplotlib honours SOURCE_DATE_EPOCH).
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

_REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = _REPO_ROOT / "memoria" / "figuras"

# ---------------------------------------------------------------------------
# Committed source artifacts (asserted to exist before anything is drawn)
# ---------------------------------------------------------------------------
MAIN_RUN = "MAIN_qrdqn_cicids2017_canonical_full_random_20260609_193655"
SRC = {
    "main_metrics": _REPO_ROOT / "runs/cicids2017" / MAIN_RUN / "metrics.json",
    "main_config": _REPO_ROOT / "runs/cicids2017" / MAIN_RUN / "config.json",
    "bootstrap": _REPO_ROOT / "runs/validation/bootstrap_ci_seed42.json",
    "duplicates": _REPO_ROOT / "runs/validation/duplicate_analysis_seed42.json",
    "check_c": _REPO_ROOT
    / "runs/validation/VAL_checks_C_20260213_004847/validation_results.json",
    "rf_random": _REPO_ROOT
    / "runs/cicids2017/baseline_random_forest_comparison"
    / "rf_cicids2017_canonical_20260628_024735__random_split/metrics.json",
    "rf_day": _REPO_ROOT
    / "runs/cicids2017/baseline_random_forest_comparison"
    / "rf_cicids2017_canonical_20260628_024735__day_split/metrics.json",
    "rf_loo": _REPO_ROOT
    / "runs/cicids2017/baseline_random_forest_comparison"
    / "rf_cicids2017_canonical_20260628_024735__leave_one_out_wednesday/metrics.json",
    "p2_metrics": _REPO_ROOT / "runs/phase2/P2v2_pred_20260610_161231_MAIN/metrics.json",
    "tb_rew": _REPO_ROOT
    / "runs/cicids2017"
    / MAIN_RUN
    / "plots/tensorboard_scalars"
    / f"{MAIN_RUN}__rollout__ep_rew_mean.csv",
    "tb_loss": _REPO_ROOT
    / "runs/cicids2017"
    / MAIN_RUN
    / "plots/tensorboard_scalars"
    / f"{MAIN_RUN}__train__loss.csv",
}
DATASET_DIR = _REPO_ROOT / "datasets/CICIDS2017"
DATASET_FILES = [  # chronological order; display labels derived from filenames only
    ("Monday-WorkingHours.pcap_ISCX.csv", "Lunes"),
    ("Tuesday-WorkingHours.pcap_ISCX.csv", "Martes"),
    ("Wednesday-workingHours.pcap_ISCX.csv", "Miércoles"),
    ("Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv", "Jueves (mañana, ataques web)"),
    ("Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv", "Jueves (tarde, infiltración)"),
    ("Friday-WorkingHours-Morning.pcap_ISCX.csv", "Viernes (mañana)"),
    ("Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv", "Viernes (tarde, PortScan)"),
    ("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", "Viernes (tarde, DDoS)"),
]

# ---------------------------------------------------------------------------
# Style (dataviz-validated palette; light/print surface)
# ---------------------------------------------------------------------------
INK = "#0b0b0b"
SEC = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"  # QRDQN / benigno (categorical slot 1)
AQUA = "#1baf7a"  # Random Forest (categorical slot 2; <3:1 contrast -> direct labels required)
RED = "#e34948"  # ataque (semantic entity color; pair CVD-validated vs BLUE)
SEQ_BLUES = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
CMAP_BLUES = LinearSegmentedColormap.from_list("thesis_blues", SEQ_BLUES)


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.labelcolor": SEC,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.9,
            "axes.titlesize": 9.5,
            "axes.titlecolor": INK,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": SEC,
            "ytick.labelcolor": SEC,
            "legend.fontsize": 8.5,
            "legend.frameon": False,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "svg.fonttype": "none",
        }
    )


def _load(key: str) -> dict[str, Any]:
    return json.loads(SRC[key].read_text(encoding="utf-8"))


def _rel(path: Path) -> str:
    return path.relative_to(_REPO_ROOT).as_posix()


def _fmt_int(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def _save(fig: plt.Figure, stem: str) -> dict[str, str]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUT_DIR / f"{stem}.pdf"
    png = OUT_DIR / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return {"pdf": _rel(pdf), "png": _rel(png)}


def _despine(ax: plt.Axes, keep: tuple[str, ...] = ("left", "bottom")) -> None:
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


# ---------------------------------------------------------------------------
# Shared confusion-matrix figure
# ---------------------------------------------------------------------------
def _cm_figure(
    stem: str,
    cells: dict[str, int],
    sources: list[Path],
    footnote: str | None = None,
    header: str | None = None,
) -> dict[str, Any]:
    tn, fp, fn, tp = cells["tn"], cells["fp"], cells["fn"], cells["tp"]
    counts = np.array([[tn, fp], [fn, tp]], dtype=float)  # rows: real benigno/ataque
    row_share = counts / counts.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(4.4, 3.5))
    ax.imshow(row_share, cmap=CMAP_BLUES, vmin=0.0, vmax=1.0)
    for i in range(2):
        for j in range(2):
            share = row_share[i, j]
            color = "white" if share >= 0.55 else INK
            ax.text(
                j, i,
                f"{_fmt_int(int(counts[i, j]))}\n({share * 100:.2f} %)",
                ha="center", va="center", color=color, fontsize=10,
            )
    ax.set_xticks([0, 1], ["Permitir", "Bloquear"])
    ax.set_yticks([0, 1], ["Benigno", "Ataque"])
    ax.set_xlabel("Acción predicha")
    ax.set_ylabel("Clase real")
    ax.tick_params(length=0)
    _despine(ax, keep=())
    if header:
        ax.set_title(header, fontsize=8.5, color=SEC, pad=10)
    if footnote:
        fig.text(0.02, -0.02, footnote, fontsize=7.5, color=MUTED, ha="left", va="top")

    files = _save(fig, stem)
    return {
        "id": stem,
        **files,
        "sources": [_rel(p) for p in sources],
        "values": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "notes": footnote or "",
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_f9_cm_main() -> dict[str, Any]:
    """F9 — confusion matrix of the MAIN model on the random test partition."""
    boot = _load("bootstrap")
    assert boot["model_verification"]["matches_recovered"] is True
    return _cm_figure(
        "f9_cm_main",
        boot["confusion_counts"],
        [SRC["bootstrap"]],
        footnote=f"Partición aleatoria de prueba, {_fmt_int(boot['n_test'])} flujos. "
        f"Ejecución {boot['run_id']}.",
    )


def fig_f11_cm_check_c() -> dict[str, Any]:
    """F11 — confusion matrix of Check C (day split). Proxy net disclosure embedded."""
    c = _load("check_c")["C"]
    return _cm_figure(
        "f11_cm_check_c",
        {"tn": c["tn"], "fp": c["fp"], "fn": c["fn"], "tp": c["tp"]},
        [SRC["check_c"]],
        header="Comprobación C — entrenamiento lun–mié, evaluación jue–vie",
        footnote=(
            f"Red proxy [512, 256] entrenada {_fmt_int(c['timesteps'])} pasos para esta "
            "comprobación; NO son los pesos del modelo MAIN. "
            f"Prueba: {_fmt_int(c['n_test'])} flujos."
        ),
    )


def fig_f14_cm_rf_dia() -> dict[str, Any]:
    """F14 — confusion matrix of the RF baseline under the day split."""
    rf = _load("rf_day")
    return _cm_figure(
        "f14_cm_rf_dia",
        {"tn": rf["tn"], "fp": rf["fp"], "fn": rf["fn"], "tp": rf["tp"]},
        [SRC["rf_day"]],
        header="Random Forest — misma partición por día que la Comprobación C",
    )


def fig_f15_fase2() -> dict[str, Any]:
    """F15 — Phase-2 offline inference on lab-captured traffic (MAIN model)."""
    m = _load("p2_metrics")
    entry = _cm_figure(
        "f15_fase2",
        {"tn": m["tn"], "fp": m["fp"], "fn": m["fn"], "tp": m["tp"]},
        [SRC["p2_metrics"]],
        header=(
            f"{_fmt_int(m['n_flows'])} flujos · tasa de bloqueo {m['block_rate'] * 100:.2f} % · "
            f"exactitud {m['accuracy']:.4f}"
        ),
        footnote=(
            "Tráfico capturado por el operador en un laboratorio doméstico cerrado; "
            "validez externa limitada."
        ),
    )
    entry["values"].update(
        {"n_flows": m["n_flows"], "block_rate": m["block_rate"], "accuracy": m["accuracy"]}
    )
    return entry


def fig_f12_duplicados() -> dict[str, Any]:
    """F12 — exact duplicates and cross-split leakage of the random partition."""
    d = _load("duplicates")
    rows = [
        ("Duplicados exactos (conjunto completo)",
         d["overall_duplicates_feature_only"]["duplicate_pct"]),
        ("Filas de prueba presentes en entrenamiento",
         d["cross_split_leakage_feature_only"]["pct_of_test"]),
        ("Ataques de prueba presentes en entrenamiento",
         d["cross_split_leakage_feature_only"]["attack_pct_of_test_attacks"]),
        ("Benignos de prueba presentes en entrenamiento",
         d["cross_split_leakage_feature_only"]["benign_pct_of_test_benigns"]),
    ]
    labels = [r[0] for r in rows][::-1]
    values = [r[1] for r in rows][::-1]

    fig, ax = plt.subplots(figsize=(6.1, 2.4))
    bars = ax.barh(labels, values, color=BLUE, height=0.62, edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, values):
        ax.text(v + 0.7, bar.get_y() + bar.get_height() / 2, f"{v:.2f} %",
                va="center", ha="left", color=SEC, fontsize=8.5)
    ax.set_xlim(0, max(values) * 1.22)
    ax.set_xlabel("Porcentaje de filas (coincidencia exacta de características)")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    _despine(ax, keep=("left",))

    files = _save(fig, "f12_duplicados")
    return {
        "id": "f12_duplicados",
        **files,
        "sources": [_rel(SRC["duplicates"])],
        "values": {r[0]: r[1] for r in rows},
        "notes": "",
    }


def fig_f13_qrdqn_vs_rf() -> dict[str, Any]:
    """F13 (star figure) — QRDQN vs Random Forest across partitions."""
    main = _load("main_metrics")
    check_c = _load("check_c")["C"]
    rf_random, rf_day, rf_loo = _load("rf_random"), _load("rf_day"), _load("rf_loo")

    groups = ["Aleatoria", "Por día", "LOO (miércoles)"]
    data = {
        "recall_attack": {
            "QRDQN": [main["recall_attack"], check_c["recall_attack"], None],
            "Random Forest": [rf_random["recall_attack"], rf_day["recall_attack"],
                              rf_loo["recall_attack"]],
        },
        "f1_attack": {
            "QRDQN": [main["f1_attack"], check_c["f1_attack"], None],
            "Random Forest": [rf_random["f1_attack"], rf_day["f1_attack"], rf_loo["f1_attack"]],
        },
    }
    panel_titles = {"recall_attack": "Recall de ataque", "f1_attack": "F1 de ataque"}

    fig, axes = plt.subplots(1, 2, figsize=(6.1, 3.1), sharey=True)
    x = np.arange(len(groups))
    width = 0.36
    for ax, metric in zip(axes, data):
        for offset, (model, color) in zip(
            (-width / 2, width / 2), [("QRDQN", BLUE), ("Random Forest", AQUA)]
        ):
            vals = data[metric][model]
            for xi, v in zip(x, vals):
                if v is None:
                    ax.text(xi + offset, 0.02, "pendiente de GPU", rotation=90,
                            ha="center", va="bottom", color=MUTED, fontsize=7.5)
                    continue
                ax.bar(xi + offset, v, width, color=color, edgecolor="white", linewidth=0.8,
                       label=model if xi == 0 else None)
                ax.text(xi + offset, v + 0.02, f"{v:.3f}", ha="center", va="bottom",
                        color=SEC, fontsize=7)
        ax.set_title(panel_titles[metric], fontsize=9, color=INK, pad=8)
        ax.set_xticks(x, groups, fontsize=8)
        ax.set_ylim(0, 1.12)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        _despine(ax, keep=("bottom",))
    axes[0].legend(loc="upper right", bbox_to_anchor=(1.02, 1.02))
    fig.text(
        0.02, -0.03,
        "QRDQN «Por día» proviene de la Comprobación C (red proxy [512, 256], 30 000 pasos), "
        "no de los pesos MAIN. QRDQN en LOO: diseñado, ejecución pendiente de GPU.",
        fontsize=7.5, color=MUTED, ha="left", va="top",
    )
    fig.tight_layout()

    files = _save(fig, "f13_qrdqn_vs_rf")
    return {
        "id": "f13_qrdqn_vs_rf",
        **files,
        "sources": [_rel(SRC[k]) for k in ("main_metrics", "check_c", "rf_random", "rf_day", "rf_loo")],
        "values": data,
        "notes": "QRDQN day-split = Check C proxy net; QRDQN LOO pending (deferred GPU register G.1).",
    }


def fig_f10_ic_bootstrap() -> dict[str, Any]:
    """F10 — 95% bootstrap CIs of the MAIN test metrics."""
    boot = _load("bootstrap")
    ci = boot["metrics_ci"]
    high = [
        ("accuracy", "Exactitud"),
        ("balanced_accuracy", "Exactitud balanceada"),
        ("mcc", "MCC"),
        ("precision_attack", "Precisión (ataque)"),
        ("recall_attack", "Recall (ataque)"),
        ("f1_attack", "F1 (ataque)"),
    ]
    rates = [("fpr", "FPR"), ("fnr", "FNR")]

    fig, axes = plt.subplots(
        1, 2, figsize=(6.1, 2.9), gridspec_kw={"width_ratios": [3, 1.4]}
    )
    for ax, metrics in zip(axes, (high, rates)):
        names = [label for _, label in metrics][::-1]
        keys = [k for k, _ in metrics][::-1]
        y = np.arange(len(keys))
        points = [ci[k]["point"] for k in keys]
        err_low = [ci[k]["point"] - ci[k]["ci95_low"] for k in keys]
        err_high = [ci[k]["ci95_high"] - ci[k]["point"] for k in keys]
        # No per-point value labels: the thesis table (tab:bootstrap-ci) is the
        # table view; the figure carries the shape of the intervals only.
        ax.errorbar(points, y, xerr=[err_low, err_high], fmt="o", color=BLUE,
                    ecolor=BLUE, elinewidth=1.4, capsize=2.5, markersize=5)
        ax.set_yticks(y, names)
        ax.set_ylim(-0.6, len(keys) - 0.4)
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        _despine(ax, keep=("bottom",))
    axes[0].set_xlabel("Valor de la métrica (IC bootstrap del 95 %)")
    axes[1].set_xlabel("Tasa de error")
    fig.text(
        0.02, -0.03,
        f"Bootstrap estratificado, {_fmt_int(boot['n_boot'])} remuestreos del conjunto de prueba "
        f"fijo ({_fmt_int(boot['n_test'])} flujos): precisión de muestreo, no varianza entre semillas.",
        fontsize=7.5, color=MUTED, ha="left", va="top",
    )
    fig.tight_layout()

    files = _save(fig, "f10_ic_bootstrap")
    return {
        "id": "f10_ic_bootstrap",
        **files,
        "sources": [_rel(SRC["bootstrap"])],
        "values": {k: ci[k] for k, _ in high + rates},
        "notes": "CIs quantify sampling precision only (single training seed).",
    }


def fig_f8_curvas_entrenamiento() -> dict[str, Any]:
    """F8 — MAIN training dynamics from the exported TensorBoard scalars."""
    import pandas as pd

    rew = pd.read_csv(SRC["tb_rew"])
    loss = pd.read_csv(SRC["tb_loss"])

    fig, axes = plt.subplots(1, 2, figsize=(6.1, 2.7))
    for ax, df, title, ylabel in (
        (axes[0], rew, "Recompensa media por episodio", "Recompensa media"),
        (axes[1], loss, "Pérdida de regresión cuantílica", "Pérdida"),
    ):
        ax.plot(df["step"] / 1e6, df["value"], color=BLUE, linewidth=1.8)
        ax.set_title(title, fontsize=9, color=INK, pad=8)
        ax.set_xlabel("Pasos de entrenamiento (millones)")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        _despine(ax)
    fig.tight_layout()

    files = _save(fig, "f8_curvas_entrenamiento")
    return {
        "id": "f8_curvas_entrenamiento",
        **files,
        "sources": [_rel(SRC["tb_rew"]), _rel(SRC["tb_loss"])],
        "values": {
            "ep_rew_mean": {"n_points": int(len(rew)), "first": float(rew["value"].iloc[0]),
                            "last": float(rew["value"].iloc[-1]),
                            "last_step": int(rew["step"].iloc[-1])},
            "train_loss": {"n_points": int(len(loss)), "first": float(loss["value"].iloc[0]),
                           "last": float(loss["value"].iloc[-1]),
                           "last_step": int(loss["step"].iloc[-1])},
        },
        "notes": "Exported from the local-only TensorBoard event log; the committed CSVs under "
        "runs/.../plots/tensorboard_scalars/ are the durable source.",
    }


def _stacked_class_bars(
    ax: plt.Axes, labels: list[str], benign: list[int], attack: list[int]
) -> None:
    y = np.arange(len(labels))
    b = np.array(benign, dtype=float) / 1e6
    a = np.array(attack, dtype=float) / 1e6
    ax.barh(y, b, color=BLUE, height=0.62, label="Benigno", edgecolor="white", linewidth=0.8)
    ax.barh(y, a, left=b, color=RED, height=0.62, label="Ataque", edgecolor="white", linewidth=0.8)
    for yi, (nb, na) in enumerate(zip(benign, attack)):
        total = nb + na
        pct = na / total * 100
        if na == 0:
            share = "sin ataques"
        elif pct < 0.05:
            share = "<0.1 % ataque"
        else:
            share = f"{pct:.1f} % ataque"
        ax.text((nb + na) / 1e6 + 0.03, yi, f"{_fmt_int(total)}  ({share})",
                va="center", ha="left", color=SEC, fontsize=7.5)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Flujos (millones)")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    _despine(ax, keep=("left",))


def fig_f7_balance_particion() -> dict[str, Any]:
    """F7 — class balance of the MAIN random partition (train/test)."""
    cfg = _load("main_config")
    sm = cfg["split_metadata"]
    labels = ["Entrenamiento", "Prueba"]
    benign = [sm["train_benign"], sm["test_benign"]]
    attack = [sm["train_attack"], sm["test_attack"]]

    fig, ax = plt.subplots(figsize=(6.1, 1.9))
    _stacked_class_bars(ax, labels, benign, attack)
    ax.set_xlim(0, (benign[0] + attack[0]) / 1e6 * 1.45)
    ax.legend(loc="lower right", ncols=2)

    files = _save(fig, "f7_balance_particion")
    return {
        "id": "f7_balance_particion",
        **files,
        "sources": [_rel(SRC["main_config"])],
        "values": {
            "train_benign": sm["train_benign"], "train_attack": sm["train_attack"],
            "test_benign": sm["test_benign"], "test_attack": sm["test_attack"],
            "train_attack_rate": sm["train_attack_rate"],
            "test_attack_rate": sm["test_attack_rate"],
        },
        "notes": "",
    }


def fig_f6_composicion_dia() -> dict[str, Any]:
    """F6 — per-day-CSV composition of CICIDS2017 (computed from the curated CSVs)."""
    import pandas as pd

    per_file: list[dict[str, Any]] = []
    for filename, label in DATASET_FILES:
        path = DATASET_DIR / filename
        col = pd.read_csv(path, usecols=lambda c: c.strip() == "Label", dtype=str).iloc[:, 0]
        col = col.str.strip()
        counts = col.value_counts().to_dict()
        n_benign = int(counts.get("BENIGN", 0))
        n_attack = int(len(col) - n_benign)
        per_file.append(
            {
                "file": filename,
                "display": label,
                "n_rows": int(len(col)),
                "n_benign": n_benign,
                "n_attack": n_attack,
                "attack_pct": round(n_attack / len(col) * 100, 4),
                "label_counts": {k: int(v) for k, v in sorted(counts.items())},
            }
        )

    fig, ax = plt.subplots(figsize=(6.1, 3.3))
    _stacked_class_bars(
        ax,
        [row["display"] for row in per_file],
        [row["n_benign"] for row in per_file],
        [row["n_attack"] for row in per_file],
    )
    ax.set_xlim(0, max(r["n_rows"] for r in per_file) / 1e6 * 1.55)
    ax.legend(loc="lower right", ncols=2)

    data_path = OUT_DIR / "data_composicion_dia.json"
    data_path.write_text(
        json.dumps(
            {
                "generated_by": "scripts/make_thesis_figures.py",
                "source": "datasets/CICIDS2017/*.csv (curated, git LFS)",
                "label_rule": "BENIGN -> benigno; any other label -> ataque",
                "per_file": per_file,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    files = _save(fig, "f6_composicion_dia")
    return {
        "id": "f6_composicion_dia",
        **files,
        "sources": [f"datasets/CICIDS2017/{f}" for f, _ in DATASET_FILES],
        "values": {r["display"]: {"n_benign": r["n_benign"], "n_attack": r["n_attack"]}
                   for r in per_file},
        "notes": f"Per-file label counts written to {_rel(data_path)} (feeds table T-E).",
    }


# ---------------------------------------------------------------------------
FIGURES = {
    "f6_composicion_dia": fig_f6_composicion_dia,
    "f7_balance_particion": fig_f7_balance_particion,
    "f8_curvas_entrenamiento": fig_f8_curvas_entrenamiento,
    "f9_cm_main": fig_f9_cm_main,
    "f10_ic_bootstrap": fig_f10_ic_bootstrap,
    "f11_cm_check_c": fig_f11_cm_check_c,
    "f12_duplicados": fig_f12_duplicados,
    "f13_qrdqn_vs_rf": fig_f13_qrdqn_vs_rf,
    "f14_cm_rf_dia": fig_f14_cm_rf_dia,
    "f15_fase2": fig_f15_fase2,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", choices=sorted(FIGURES), default=None)
    parser.add_argument("--skip-eda", action="store_true",
                        help="Skip f6_composicion_dia (full dataset scan).")
    args = parser.parse_args()

    missing = [str(p) for p in SRC.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing source artifacts: {missing}")

    selected = args.only or list(FIGURES)
    if args.skip_eda and "f6_composicion_dia" in selected and not args.only:
        selected.remove("f6_composicion_dia")

    _apply_style()
    manifest_path = OUT_DIR / "figures_manifest.json"
    manifest: dict[str, Any] = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {"figures": {}}
    )
    for name in selected:
        entry = FIGURES[name]()
        manifest["figures"][name] = entry
        print(f"[ok] {name} -> {entry['pdf']}")

    manifest["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest["generator"] = "scripts/make_thesis_figures.py"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
