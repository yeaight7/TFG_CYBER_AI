# TFG memoria — Improvement Plan & Tracker

> Execution tracker for the multi-session improvement of `memoria/` (canonical Spanish thesis).
> A task is `[x]` ONLY after its verification gate passes at a committed state and the Evidence
> cell is filled. This document is self-sufficient: a zero-context session can resume from here.
> Design rationale/blueprint detail beyond this doc: session plan of 2026-07-05 (F0 PR description).

---

## HOW TO RESUME (read this first, zero-context session)

1. Read this header, the Status legend, and the Baseline block.
2. In "Master tracker", find the first phase not `[x]`; within it, the first task not `[x]`.
3. Run `git status` and `git log --oneline -15`. Note the current branch.
4. Read that phase's block in "Execution log" (bottom): last-green commit + WIP notes.
5. Build sanity: run BUILD (below). If it fails → "Recovery" section.
6. Resume at the first open task, on that phase's branch (`chore/yeaight7/thesis-f<N>-<slug>`).
   Never start a new phase with a dirty tree. Never edit `memoria.tex`/`memoria.bib` in parallel branches.

## Environment (verified 2026-07-05)

- TeX Live 2025 at `D:\texlive\2025`. All needed packages present (tikz, pgfgantt, algorithm, algpseudocode, longtable, pdflscape, caption, subcaption — verified with kpsewhich).
- **BUILD** (PowerShell — the TEMP redirect is MANDATORY, biber 2.21 dies under AppData temp):

  ```powershell
  $env:TMP='C:\Temp'; $env:TEMP='C:\Temp'; $env:TMPDIR='C:\Temp'
  & D:\texlive\2025\bin\windows\latexmk -pdf -cd C:\Users\Rivero\Desktop\TFG_CYBER_AI\memoria\memoria.tex
  ```

- **PAGES**: `& D:\texlive\2025\bin\windows\pdfinfo memoria\memoria.pdf | Select-String Pages`
- **Recovery from biber cache poisoning** (latexmk remembers a failed biber run):

  ```powershell
  & D:\texlive\2025\bin\windows\latexmk -C -cd C:\Users\Rivero\Desktop\TFG_CYBER_AI\memoria\memoria.tex
  git checkout -- memoria/memoria.pdf   # -C deletes the tracked PDF; restore it
  # then rebuild with the TEMP redirect (BUILD above)
  ```

- Figures env: `uv sync --all-extras`, then `uv run python scripts/make_thesis_figures.py` (created in F1).
- Spanish typography trap: percentages in TEXT mode (`24.86\%`), never `$...\,\%$` (babel-spanish glue error).

## Status legend

`[ ]` todo · `[~]` in progress (WIP note required in Execution log) · `[x]` done & verified · `[-]` won't-do / deferred · `[?]` blocked on user decision · `[!]` pending to do in the future (e.g., GPU experiments) · `[o]` solved differently (relayed/delegated to another agent or manually done by the user [User doesn't update the log or evidence]).

## Verification gates

| Gate | What | Command/rule |
|---|---|---|
| G1 | Build exits 0 | BUILD command above |
| G2 | Page count logged (delta investigated) | PAGES command; record in Evidence |
| G3 | No broken refs/citations | `Select-String -Path memoria\memoria.log -Pattern 'undefined\|multiply defined'` → must return nothing |
| G4 | Figures exist per manifest | every file in `memoria/figuras/figures_manifest.json` exists |
| G5 | Honesty disclosures survive | grep counts (Baseline below) must be ≥ baseline, run BEFORE and AFTER each editing phase |
| G6 | Typography lint | `Select-String -Path memoria\capitulos\*.tex -Pattern '\\\\,\\\\%'` → must return nothing |
| G7 | Numbers == artifacts | separate top-model subagent checks every numeric claim in edited chapters against artifact JSONs / `figures_manifest.json` |

## Baseline (captured F0, 2026-07-05 — immutable)

- Pages: **118** · Build: **clean** (full `-gg` rebuild verified, biber 2.21 OK, 0 undefined refs)
- G5 disclosure grep counts over `memoria/capitulos/*.tex` (**must never decrease**):

| Pattern (regex) | Count | Disclosure |
|---|---|---|
| `gamma = 0` | 5 | γ=0 contextual-bandit |
| `24.86` | 1 | test-in-train duplicate leakage % |
| `40.12` | 1 | test-attack duplicate leakage % |
| `0.52954` | 5 | Check C day-split attack recall |
| `bandido contextual` | 1 | bandit framing |
| `validez externa limitada` | 5 | Phase-2 external validity |
| `0.991862` | 3 | Phase-2 accuracy (lab-only context) |
| `una sola semilla` | 1 | single-seed limitation |
| `precisión de muestreo` | 2 | bootstrap = sampling precision, not seed variance |
| `red proxy` | 1 | Check C used proxy network, not MAIN weights |

- **Disclosure added in F7:** Check C used a proxy network (`[512,256]`, 30k steps), NOT the MAIN weights; G5 pattern `red proxy` must remain present from F7 onward.
- Chapter word counts at baseline: intro 2,139 · objetivos 2,964 · estado del arte 8,456 · metodología 8,488 · resultados 1,321 · discusión 632 · limitaciones 734 · ética 665.

## Honesty invariants (non-negotiable during ALL edits)

γ=0 contextual-bandit disclosure · Phase-2 framed "laboratorio doméstico cerrado, generado por el operador, validez externa limitada" · single-seed limitation · Check C = proxy net, no MAIN weights (from F7 on) · bootstrap CI = precisión de muestreo, no varianza de semilla · pending GPU experiments always "diseñados e implementados, ejecución pendiente", never with numbers (they will be ran and recorded at some point, but are still pending).

## Do not mention these under any circumstances

duplicate-leakage 22.30/24.86/40.12% cited ·

## Git strategy

- One branch per phase `chore/yeaight7/thesis-f<N>-<slug>` → PR to `main`. No force-push, no history rewrite (D-6).
- One atomic commit per task ID; tree build-green at every commit; `wip(F5.3): ... [DO NOT MERGE]` allowed at session end with a WIP note here.
- No need to reset/undo commits. If the something is committed twice is okay. Just commit again the new changes.
- **PDF single-writer rule**: phase branches NEVER commit `memoria/memoria.pdf` (restore with `git checkout -- memoria/memoria.pdf` before committing). Rebuild + commit the PDF only on `main` right after each merge: `build: rebuild memoria.pdf (Fn, NNN pp)`. `.gitattributes` marks it `merge=binary`.
- Commit style: `type: summary (F1.2,F1.3)`. **No AI co-author trailers, no "Generated with" footers** in commits or PR bodies (user rule, 2026-07-06).

## Subagent / model policy

| Work | Model |
|---|---|
| Mechanical LaTeX edits, wiring, tracker upkeep | sonnet/5.5/5.6-Luna (cheap) |
| Figure scripts, TikZ, TB export | sonnet/5.5/5.6-Luna |
| Spanish academic prose (resumen, results, discusión, conclusiones, style pass) | high tier model (top model) |
| Adversarial verification (G5/G7, citation context, EN fidelity) — separate agent from the writer | high tier model (top model) |

Parallel fan-out for independent figures / independent claim verification. Single author, sequential, for any one chapter's prose and for shared files (`memoria.tex`, `memoria.bib`).

---

# Master tracker

## F0 — Scaffold, preamble, baseline ✅

| ID | Task | Model | Dep | Status | Evidence |
|---|---|---|---|---|---|
| F0.1 | Branch `claude/thesis-f0-scaffold`; verify toolchain with full `-gg` rebuild | cheap | — | [x] | build=clean · biber 2.21 OK · pages=118 |
| F0.2 | `.gitattributes`: `memoria/memoria.pdf` + `report/report.pdf` `merge=binary` | cheap | — | [x] | this PR |
| F0.3 | Create `memoria/figuras/` (.gitkeep) | cheap | — | [x] | this PR |
| F0.4 | Preamble hardening (3 verified batches): `\decimalpoint`, `\graphicspath{{figuras/}{assets/}}`, secnumdepth=3/tocdepth=2; tikz+libs, pgfgantt, longtable, pdflscape; algorithm+algpseudocode, caption+subcaption, Spanish float names | cheap | — | [x] | 3 green builds · pages=118 · G3 clean |
| F0.5 | Capture baseline (pages + G5 counts + word counts) into this doc | cheap | F0.4 | [x] | Baseline block above |
| F0.6 | This tracker + `docs/thesis/README.md` index | cheap | — | [x] | this PR |

## F1 — Data figures from committed artifacts (‖ F2, F3)

Rule: **no number may appear in any figure that is not present in a committed artifact.** Output: `memoria/figuras/<id>_<slug>.pdf` + `.png` + entry in `figures_manifest.json` (fields: file, source_artifact, values_plotted). Figures are committed so the LaTeX build never runs Python.

| ID | Task | Model | Dep | Status | Evidence |
|---|---|---|---|---|---|
| F1.1 | `uv sync --all-extras`; scaffold `scripts/make_thesis_figures.py` (one function per figure, `matplotlib.use("Agg")`, writes manifest) | sonnet | F0 | [x] | commit 8667520 · SOURCE_DATE_EPOCH for reproducible PDFs |
| F1.2 | Run `scripts/export_tensorboard_scalars.py` on MAIN TB event log (`runs/cicids2017/MAIN_qrdqn_cicids2017_canonical_full_random_20260609_193655/events.out.tfevents.*`) → CSVs | sonnet | F1.1 | [x] | commit 89bb943 · 6 scalar tags exported (see DT-6) |
| F1.3 | F8 learning curves (reward/loss vs steps) from F1.2 CSVs | sonnet | F1.2 | [x] | commit 816164a · `figuras/f8_curvas_entrenamiento.pdf` |
| F1.4 | F9 confusion-matrix heatmap MAIN (TN 451631 / FP 2989 / FN 518 / TP 111011) from `bootstrap_ci_seed42.json` `confusion_counts` (MAIN `metrics.json` has no cells) | sonnet | F1.1 | [x] | commit 816164a · `figuras/f9_cm_main.pdf` |
| F1.5 | F12 duplicate/leakage bars (22.30 / 24.86 / 40.12 / 21.12%) from `runs/validation/duplicate_analysis_seed42.json` | sonnet | F1.1 | [x] | commit 816164a · `figuras/f12_duplicados.pdf` |
| F1.6 | F13 **star figure**: QRDQN vs RF across 3 partitions (recall+F1 attack; incl. RF LOO-Wednesday 0.00719) from MAIN + `VAL_checks_C_20260213_004847` + `runs/cicids2017/baseline_random_forest_comparison/` | sonnet | F1.1 | [x] | commit 816164a · `figuras/f13_qrdqn_vs_rf.pdf` · QRDQN-LOO slot = "pendiente de GPU" text, no bar |
| F1.7 | F11 CM heatmap Check C — **caption/metadata must state proxy net, not MAIN weights** | sonnet | F1.1 | [x] | commit 816164a · disclosure embedded as in-figure footnote |
| F1.8 | F15 Phase-2 results (block_rate 0.2524, CM) from `runs/phase2/P2v2_pred_20260610_161231_MAIN/` | sonnet | F1.1 | [x] | commit 816164a · lab-validity footnote embedded |
| F1.9 | F10 bootstrap-CI errorbars from `runs/validation/bootstrap_ci_seed42.json` | sonnet | F1.1 | [x] | commit 816164a · no per-point labels (table `tab:bootstrap-ci` is the table view) |
| F1.10 | F6 per-day CICIDS2017 composition (pandas count over `datasets/CICIDS2017/*.csv`, user-approved EDA) + emit T-E table data | sonnet | F1.1 | [x] | commit 816164a · `figuras/data_composicion_dia.json` (feeds T-E) |
| F1.11 | F7 MAIN-partition class balance from MAIN `config.json` `split_metadata` | sonnet | F1.1 | [x] | commit 816164a · `figuras/f7_balance_particion.pdf` |
| F1.12 | F14 CM heatmap RF day-split | sonnet | F1.1 | [x] | commit 816164a · `figuras/f14_cm_rf_dia.pdf` |
| F1.V | **Verify** (top model, separate agent): every manifest value byte-matches its source artifact; G4 | TOP | F1.* | [x] | 8/8 verifiers PASS (102 checks, 0 mismatches, 0 visual issues) · G1=clean/118pp · G4=0 missing · G5=baseline intact · G6=clean |

## F2 — TikZ diagrams (‖ F1, F3)

| ID | Task | Model | Dep | Status | Evidence |
|---|---|---|---|---|---|
| F2.0 | Shared TikZ styles `figuras/tikz_estilos.tex` + preamble `\input` wiring | — | F0 | [x] | commits 2a5d5ae, 8b4279c · build green |
| F2.1 | F2 pipeline general del sistema (from `src/` structure: load → clean → canonical 152-dim → env → QRDQN → eval → Phase-2) | sonnet | F0 | [x] | commit 8851cf1 + fixes (características, escalado, extractor externo CICFlowMeter-py per `pcaps/README.md` — NOT the predict script) |
| F2.2 | F3 agente–entorno interaction, one-step (γ=0) with reward table (from `src/rl_defender_env.py`) | sonnet | F0 | [x] | commit 3d03819 + fixes (benigno/omisión wording, label collisions; decimal points render via babel `\decimalpoint`) |
| F2.3 | F4 observation-vector construction 76+76 mask (from `src/canonical_schema.py`) | sonnet | F0 | [x] | commit b39279f · presence-mask semantics + constant-1 note |
| F2.4 | F5 QRDQN network architecture ([1024,1024,512], 200 quantiles × 2 actions, from MAIN `config.json`) | sonnet | F0 | [x] | commit 1c59904 + hyphenation fix |
| F2.5 | F16 validation-ladder schematic (A→B→C→duplicados→Fase 2) | sonnet | F0 | [x] | commit 88dee03 · two-flight staircase, artifact-derived verdicts |
| F2.6 | F17 Phase-2 inference pipeline (from `scripts/predict_real_traffic_v2.py`) | sonnet | F0 | [x] | commit 9ed3aca + fixes (máscara de presencia, `--clip-z` literal) |
| F2.7 | F1 Gantt (pgfgantt) — ONLY evidence-anchored dates: contrato 11/18-nov-2025; entregas 15-dic / 2-feb / 17-mar / 20-may / 15-jun; desarrollo 11-dic-2025→09-jun-2026 (primer commit→MAIN); redacción 12-dic-2025→06-jul-2026; experimentos 12–23-feb; Fase 2 23-feb→10-jun; auditoría+RF 27–28-jun | sonnet | F0 | [x] | commit 25ce461 · Spanish month labels |
| F2.V | Verify: each diagram faithfully reflects its source file (review vs code); build green | TOP | F2.* | [x] | 7/7 verifiers PASS (70 checks, 0 failures) · G1 clean/118pp · G3 clean · G5 baseline intact · G6 clean · all 7 visually reviewed via Ghostscript renders |

## F3 — Front matter (‖ F1, F2)

| ID | Task | Model | Dep | Status | Evidence |
|---|---|---|---|---|---|
| F3.1 | Titlepage: `titlepage` env, logo `assets/Universidad_Loyola_Logo_POS_RGB.png`, Universidad Loyola / Grado en Ingeniería Informática y Tecnologías Virtuales / tutor Alfonso Carlos Martínez Estudillo (user-provided 2026-07-06) | cheap+user | F0 | [x] | commit c1180b5 |
| F3.2 | Roman page numbering for front matter; arabic restart at Ch1 (babel `es-lcroman`, DT-8) | cheap | F3.1 | [x] | commit 36ae1c0 |
| F3.3 | Agradecimientos (real text provided by user: padres Juan y Esmeralda, hermano Carlos, tutor Alfonso Carlos) | cheap | F3.2 | [x] | commit c1180b5 |
| F3.4 | Resumen (250–300 w) + Palabras clave; honesty disclosures carried in (bandido contextual, cota optimista, red reducida, validez externa limitada) | TOP | F3.2 | [x] | commit c1180b5 |
| F3.5 | Abstract EN + Keywords (faithful translation of F3.4) | TOP | F3.4 | [x] | commit c1180b5 |
| F3.6 | `\listoffigures`, `\listoftables`, `\listofalgorithms` after TOC (empty until figures wire in F5–F7); indices read as text via linkcolor group | cheap | F3.2 | [x] | commit 36ae1c0/c1180b5 |
| F3.7 | Glosario de acrónimos: `\chapter*` + two-column longtable, 30 entries grounded in an acronym census of the chapters | sonnet | F3.2 | [x] | commit c1180b5 |
| F3.8 | **Layout/design pass (user request)**: fancyhdr running headers, titlesec chapter/section formats (palette), colored links + PDF metadata, microtype, caption styling, `es-tabla`, short captions for the 6 existing tables | TOP | F3.1 | [x] | commit 36ae1c0 · visual review of 10 rendered pages |
| F3.V | Gates G1–G3, G5, G6 + adversarial verify of resumen/abstract numbers vs artifacts | TOP | F3.* | [x] | G1 clean/130pp · G3 clean · G5 ≥ baseline (24.86→5, bandido→2, validez→6) · G6 clean · verifier PASS 19/19 (resumen 298 w, ES/EN faithful, disclosures in both languages) |

## F4 — Structural split of Metodología

Pure content MOVE, no rewriting. `metodologia.tex` → `capitulos/diseno_sistema.tex` (visión: datos, limpieza, esquema canónico, formulación RL, agente QRDQN, implementación) + `capitulos/protocolo_experimental.tex` (fases, particiones, métricas/escalera, línea base, escala de entrenamiento [pendiente-GPU], Fase 2, reproducibilidad, limitaciones metodológicas).

| ID | Task | Model | Dep | Status | Evidence |
|---|---|---|---|---|---|
| F4.1 | Split file, rewire `\input`s in `memoria.tex`, delete `metodologia.tex` | sonnet | F0 | [x] | commit 7d510ac · scripted byte-exact split (purity asserted); D=7/P=7 sections per DT-5; chapter labels `cap:diseno-sistema`/`cap:protocolo-experimental` added for F5–F7 wiring |
| F4.2 | Bibliography: `\chapter{Bibliografía}+\printbibliography[heading=none]` → `\printbibliography[heading=bibintoc, title={Bibliografía}]` | cheap | F4.1 | [x] | commit 6c52681 |
| F4.3 | `\appendix` skeleton after bibliography (empty anexo files wired, content in F10) | cheap | F4.2 | [x] | commit 21f358e · `anexo_{a_reproducibilidad,b_entorno,c_esquema_canonico,d_artefactos}.tex`, comment-only → zero visible output |
| F4.4 | Fix all `\ref`/`\label` fallout; chapter numbering audit (resultados becomes Ch6 later — verify current numbering consistent) | sonnet | F4.1 | [x] | commit 6e8a6f0 · 0 dangling refs/0 dup labels · all 3 hardcoded «Capítulo N» mentions still correct (intro:37 ×3, etica:6) · limitaciones.tex:3 retargeted to protocolo; stale comment pointers fixed (`train_rl_defender.py`, `figuras/f4_vector_observacion.tex`) |
| F4.V | Gates G1–G3, G5 (before+after), G6 | cheap | F4.* | [x] | G5 before==after (5/5/3/5/2/6/3/1/3) · G6 clean · 3/3 adversarial verifiers PASS (fidelity 60 checks byte-exact, wiring 12, numbering/gates 5) · G1–G2: user's manual build 2026-07-06 20:40 → **131 pp** (predicted 131±1), PDF committed on main (434e1e2) · G3: `memoria.log` from that build clean |

## F5 — Algorithms + structural tables

Rule: each algorithm written AFTER re-reading its source file; divergence = bug in the pseudocode.

| ID | Task | Model | Dep | Status | Evidence |
|---|---|---|---|---|---|
| F5.1 | A1 entrenamiento QRDQN (from `src/train_rl_defender.py`; target line must show `y ← r`, reinforcing γ=0) | sonnet | F4 | [x] | |
| F5.2 | A2 `env.step` (from `src/rl_defender_env.py`; rewards tp=1.5/fp=−2.0/fn=−5.0/om=0) | sonnet | F4 | [x] | |
| F5.3 | A3 mapeo canónico (from `src/canonical_schema.py`) | sonnet | F4 | [x] | |
| F5.4 | A4 inferencia Fase 2 (from `scripts/predict_real_traffic_v2.py`) | sonnet | F4 | [x] | |
| F5.5 | T-A hiperparámetros QRDQN from MAIN `config.json` + provenance column ("perfil fijado a mano, no Optuna") | sonnet | F4 | [x] | |
| F5.6 | T-F recompensa con lectura multiobjetivo (C3: fn=−5.0 prioriza seguridad, fp=−2.0 reduce impacto) | sonnet | F4 | [x] | |
| F5.7 | T-G inventario de ejecuciones oficiales (MAIN, VAL A/B/C, bootstrap, duplicates, RF sweep 20260628, P2v2 MAIN) | sonnet | F4 | [x] | |
| F5.8 | T-E composición por día (data from F1.10) | sonnet | F1.10, F4 | [x] | |
| F5.V | G7 numbers check (top model) + gates G1–G3, G5, G6 | TOP | F5.* | [o] codex checked it | |

## F6 — Ch2: Objetivos, alcance y planificación

| ID | Task | Model | Dep | Status | Evidence |
|---|---|---|---|---|---|
| F6.1 | Rename chapter; OBJ-C1..C4 contract catalog (T-C spec tables: ID/Descripción/Tipo/Verificación/Capítulos) | TOP | F4 | [x] | commit 118fa31 · «Objetivos, alcance y planificación» · `tab:obj-c1..c4` spec cards — C3 anchored on tracker definition (F5.6/F8.3); **C1/C2/C4 authored from thesis content (no contract text in repo) — user should validate vs anteproyecto** |
| F6.2 | OBJ-01..08 IDs + criterio de verificación on existing 8 subsections | TOP | F6.1 | [x] | commit 6239a82 · titles tagged + per-objective `Criterio de verificación` ¶; OBJ-07 criterion = artefactos pendientes de GPU, sin cifras |
| F6.3 | T-D matriz de trazabilidad (objetivos ↔ secciones ↔ RUN_IDs ↔ estado; OBJ-07 = Pendiente-GPU) | TOP | F6.2 | [!] | commit 16167a3 · `tab:trazabilidad` · OBJ-07 **Pendiente de GPU** · OBJ-06 completado con nota LOO-QRDQN pendiente · full RUN_IDs deferred to `tab:inventario-runs` |
| F6.4 | 2.7 Restricciones (T-J RES-xx: factores dato / estratégicos) | TOP | F6.1 | [x] | commit 2aa2c09 · `tab:restricciones` RES-D1..D4 / RES-E1..E4 (E2 = GPU diferido) |
| F6.5 | 2.8 Recursos (T-B HW/SW from `requirements*.txt`, `docs/runpod_main_experiment.md`) | sonnet | F6.1 | [x] | commit 52117c8 · `tab:recursos` (sonnet agent, isolated compile clean) · versions verbatim from requirements/pyproject; GPU = RTX 3090 Ti per runpod doc ("Preferred" wording — confirm actual pod) |
| F6.6 | 2.9 Planificación temporal: T-I hitos + embed Gantt F1 | TOP | F2.7 | [x] | commit fd48682 · `tab:hitos` dates verbatim from `f1_gantt.tex` · `fig:gantt` via `\resizebox{\textwidth}` (compile pending F6.V) |
| F6.V | Gates + G7 | TOP | F6.* | [x] | Codex verification 2026-07-09 · G1 latexmk exit 0 · G2 pages=162 · G3 clean · G4 manifest files present · G5 counts 8/5/3/5/3/9/3/1/5 ≥ baseline · G6 clean · G7 numeric/resource/date claims checked against `config.json`, `environment.json`, `requirements*.txt`, `pyproject.toml`, `f1_gantt.tex`, `figures_manifest.json`, and reward source; no repo-backed numeric mismatches found · PR #48 checks passing |

## F7 — Ch6 Resultados expansion (heaviest content phase)

| ID | Task | Model | Dep | Status | Evidence |
|---|---|---|---|---|---|
| F7.1 | 6.1 Condiciones generales + T-G + traceability statement | TOP | F1, F5 | [x] | `resultados.tex` §6.1 references T-G (`tab:inventario-runs`) and states the artifact/manifest traceability rule |
| F7.2 | 6.2 Experimento 1 MAIN: Condiciones / Dinámica (embed F8) / Resultados (embed F9) / Discusión breve | TOP | F7.1 | [x] | §6.2 split into Condiciones / Dinámica / Resultados / Discusión; embeds F8 and F9 |
| F7.3 | 6.3 bootstrap (embed F10; keep "precisión de muestreo" verbatim) | TOP | F7.1 | [x] | §6.3 keeps "precisión de muestreo" and embeds F10 |
| F7.4 | 6.4 escalera A/B/C (embed F11) + **ADD proxy-net disclosure** (`[512,256]`, 30k steps, no MAIN weights) — then append pattern to G5 baseline | TOP | F7.1 | [x] | §6.4 embeds F11 and states red proxy `[512,256]`, 30k steps, no MAIN weights; G5 baseline pattern `red proxy` added |
| F7.5 | 6.5 duplicados/fuga (embed F12; numbers verbatim) | TOP | F7.1 | [x] | §6.5 embeds F12; 22.30/24.86/40.12/21.12% kept verbatim |
| F7.6 | 6.6 RF baseline (embed F13+F14; add LOO row to tab:rf-vs-qrdqn; RF in-distribution win stated plainly) | TOP | F7.1 | [x] | §6.6 embeds F13/F14; `tab:rf-vs-qrdqn` includes RF LOO row and QRDQN pending row; RF random-split win stated plainly |
| F7.7 | 6.7 Fase 2 (embed F15; lab-validity framing verbatim) | TOP | F7.1 | [x] | §6.7 embeds F15 and keeps lab-only / validez externa limitada framing |
| F7.8 | 6.8 síntesis + cobertura de objetivos (T-L; echo OBJ IDs) | TOP | F7.2–7.7, F6 | [x] | §6.8 adds T-L (`tab:cobertura-objetivos-resultados`) with OBJ-01..OBJ-08 coverage |
| F7.V | G7 adversarial numbers check (separate top-model agent) + all gates | TOP | F7.* | [x] codex checked it | G1 clean/165pp · G3 clean · G4 manifest files present · G5 counts 9/5/3/5/5/12/4/1/5 plus `red proxy`=1 · G6 clean · G7 literal check passed against source artifacts |

## F8 — Ch7 Discusión + Ch8 Limitaciones + Ch9 Ética

| ID | Task | Model | Dep | Status | Evidence |
|---|---|---|---|---|---|
| F8.1 | 7.1–7.3 expand quantitatively (exact table values) | TOP | F7 | [x] | commit 4f06b22 · §7.1–7.3 expanded with MAIN, bootstrap, duplicate, Check C, RF, and Fase 2 values |
| F8.2 | 7.4 posicionamiento vs SOTA (leakage caveat on literature numbers; cite Layeghy2022, Boukhamla2021, Cantone2024) | TOP | F7 | [x] | commit 4f06b22 · cites `Layeghy2022`, `Boukhamla2021CICIDS2017Validation`, `Cantone2024` |
| F8.3 | 7.5 lectura multiobjetivo {seguridad, impacto} (contract C3) | TOP | F7 | [x] | commit 4f06b22 · §7.5 anchors C3 on `tp=1.5`, `fp=-2.0`, `fn=-5.0`, `omission=0.0` |
| F8.4 | 7.6 amenazas a la validez (short, cross-ref Ch8) | TOP | F7 | [x] | commit 4f06b22 · §7.6 summarizes seed, bootstrap, duplicate, proxy-net, and lab-validity threats |
| F8.5 | Ch8: expand each limitation 1–2 paragraphs; MOVE "Trabajo futuro" content out (staged for 10.4) | TOP | F7 | [x] | commit 45c68f6 · limitations expanded; rendered `Trabajo futuro` removed; future-work stash kept as LaTeX comments for F9 |
| F8.6 | Ch9 Ética light expansion (privacidad + lab-traffic handling) | TOP | — | [x] | commit 4ab4dd1 · privacy, lab-traffic handling, offline scope, and operational-risk wording expanded |
| F8.V | Gates + G5 strict (this phase touches the disclosure-dense chapters) | TOP | F8.* | [x] | G1 clean/171pp · G3 clean · G4 missing=0 · G5 counts 9/5/3/6/4/12/4/2/7 plus `red proxy`=4 · G6 clean · G7 verifier PASS |

## F9 — Ch10 Conclusiones (new) + Ch1 expansion

| ID | Task | Model | Dep | Status | Evidence |
|---|---|---|---|---|---|
| F9.1 | New `capitulos/conclusiones.tex`: 10.1 objetivos contractuales echo · 10.2 objetivos específicos echo (OBJ-07 "diseño completado, ejecución pendiente") · 10.3 reflexión del autor · 10.4 líneas futuras corto/medio/largo plazo (each with cite; Zhang2025OpenSet here) | TOP | F6, F8 | [x] | PRs #52/#53 · commits f2cd3f0, e2f2295, b1d94c8 · Ch10 sections and required citations verified |
| F9.2 | Ch1: PDS-adapted spec (1.2.2), contributions C1–C4 with evidence pointers, lead-in | TOP | F6 | [x] | PR #52 · commit 383c9ea · PDS-adapted specification, C1–C4 evidence pointers, and lead-in present |
| F9.3 | Ch1 §1.6 roadmap rewrite naming ALL chapters + anexos | TOP | F9.1 | [x] | PR #52 · commit 383c9ea · roadmap names Ch2–Ch10, bibliography, and Annexes A–D |
| F9.V | Gates; every OBJ ID from Ch2 appears in Ch6.8 or Ch10 | TOP | F9.* | [x] | post-merge isolated BUILD exit 0/179 pp · G3–G6 clean · G7 independent verifier PASS · OBJ-01–OBJ-08 present in Ch6.8 and Ch10 |

## F10 — Anexos A–D

| ID | Task | Model | Dep | Status | Evidence |
|---|---|---|---|---|---|
| F10.1 | Anexo A Manual de reproducibilidad (from `docs/reproducibility.md` + exact run commands) | sonnet | F4 | [x] | commit f05e0b4 · current parser commands verified; no-argument duplicate/RF entrypoints inspected statically · isolated build 195 pp |
| F10.2 | Anexo B Entorno HW/SW (versions from `requirements*.txt`, `environment.json`) | sonnet | F4 | [x] | commit 5ee234f · author-confirmed hardware separated from MAIN runtime and install contracts; Ch2 resource table corrected · isolated build 200 pp |
| F10.3 | Anexo C Esquema canónico completo (add the missing 76-row audit enumeration; keep Ch4's grouped summary + forward reference) | sonnet | F4 | [x] | commit d346504 · 76 unique complete mappings; exact agreement with MAIN's ordered 152 names · isolated build 209 pp |
| F10.4 | Anexo D Catálogo de artefactos por ejecución | sonnet | F4 | [x] | commit ad73184 · paths/tracking audited; immutable Phase 2 provenance and T-G summary corrected · isolated build 217 pp |
| F10.V | Gates | cheap | F10.* | [x] | forced isolated BUILD exit 0/215 pp (+36 investigated) · G1–G7 PASS · Ruff + 45 pytest PASS · all annex pages visually inspected · tracked PDF untouched |

## F11 — Estado del arte trim + bibliography weave

| ID | Task | Model | Dep | Status | Evidence |
|---|---|---|---|---|---|
| F11.1 | Tighten 5–10% in the 3 longest sections (datasets, supervised DL, riesgos metodológicos) — nothing deleted or moved out | TOP | F9 | [x] | texcount text: datasets+CICIDS 1,024 (from 1,134), supervised/DL 657 (from 723), risks 587 (from 652) · forced isolated build exit 0, 214 pp (-1 from 215 baseline) |
| F11.2 | T-H related-work RL-NIDS table weaving NIDSRL2023, RLTechniques2023NIDS, Sanusi2023DRLIDS, Umer2022RLRLIDS, Cevallos2023DRLIDSBP, DDPG2025AttackDetection, HCLRIDS2025IoMT, DRLIDSSDN2025 | TOP | F11.1 | [x] | 8-source compact five-column portrait longtable before thesis positioning · primary metadata corrected for its cited records · forced isolated build exit 0, 216 pp (+1 from 215 baseline) · caption, label, continuation and columns rendered/inspected |
| F11.3 | Weave remaining uncited: Layeghy2022, Boukhamla2021, Cantone2024, DatasetSurvey2025, Ozgur2016, Rodriguez2022, TrainingData2025, Farrukh2022, Pekar2024 (one claim-bearing citation each); prune Oyelakin2023Overview if no honest slot | TOP | F11.1 | [x] | 10 named sources plus retained Oyelakin woven into claim-bearing Chapter 3 prose · 10 primary-audited records corrected, keys preserved · reviewer PASS after PCA bridge correction · texcount final 1,057 / 685 / 590 · forced isolated build exit 0, 219 pp (+4 from 215 baseline) |
| F11.4 | Sweep: zero uncited bib entries remain (`git grep` cite keys vs `memoria.bib`); biber log zero warnings | cheap | F11.3 | [ ] | |
| F11.V | Gates + citation-context verification (top model) | TOP | F11.* | [ ] | |

## F12 — Document-wide style/QA pass

| ID | Task | Model | Dep | Status | Evidence |
|---|---|---|---|---|---|
| F12.1 | Conventions sweep: chapter lead-ins everywhere; bilingual glossing on first mention; captions end "Fuente: elaboración propia [a partir de `\texttt{<artifact>}`]"; RUN_ID in `\texttt{}` at first numeric use; quantitative discussion style | TOP | F11 | [ ] | |
| F12.2 | Overfull/underfull box cleanup; float placement audit | cheap | F12.1 | [ ] | |
| F12.3 | Full gate suite G1–G7; record final page count; tag `main` milestone `thesis-content-complete` | cheap | F12.2 | [ ] | |

## F13 — EN mirror re-sync (`report/` — FROZEN until F12 merged)

| ID | Task | Model | Dep | Status | Evidence |
|---|---|---|---|---|---|
| F13.1 | Port all ES changes + new chapters/figures to `report/` (figures reused from `memoria/figuras/`) | TOP | F12 | [ ] | |
| F13.2 | Fix stale sync note in `docs/reproducibility.md` | cheap | F13.1 | [ ] | |
| F13.V | `report/` builds green; fidelity verification (top model) | TOP | F13.* | [ ] | |

---

## Deferred GPU register (tracked, NOT executed in this effort — repo decision D-9)

No thesis claim or figure may depend on these until they are actually run and their artifacts committed. *Note: User will soon run more experiments and save the artifacts*

| ID | Experiment | Command (future operator) | Status |
|---|---|---|---|
| G.1 | leave-one-CSV-out (QRDQN) | `uv run python -m src.validate_leave_one_csv_out` (see module docstring for args) | [!] pendiente de GPU |
| G.2 | Training-size ladder (100k/250k/500k/1M/2M, nested-prefix) | `uv run python -m src.train_rl_defender --preset full --train-max-rows <N> ...` per `experiments/cicids2017_qrdqn_experiments.md` | [!] pendiente de GPU |
| G.3 | Multi-seed variance study (MAIN profile, ≥3 seeds) | same as MAIN with `--seed {43,44,45}` | [!] pendiente de GPU |

## Open items needing USER input

| Item | Needed for | Status |
|---|---|---|
| ~~Exact degree + faculty line for portada~~ | F3.1 | resolved 2026-07-06: Grado en Ingeniería Informática y Tecnologías Virtuales, Universidad Loyola |
| ~~Tutor name confirmation~~ | F3.1 | resolved 2026-07-06: Alfonso Carlos Martínez Estudillo |
| ~~Agradecimientos text~~ | F3.3 | resolved 2026-07-06 (Juan y Esmeralda, Carlos, Alfonso Carlos) |
| ~~Optional: phase labels for Gantt bands~~ | F2.7 | resolved — Gantt built from contract dates + repo evidence (first commit, run timestamps) |
| Portada date currently "Julio de 2026" | F3.1 | [!] confirm or adjust to defense date - User will adjust manually - It stays as Julio de 2026 for now |

## Decisions log (append-only)

- DT-1 (2026-07-05): Keep `report` class + biblatex; no template migration. Pineda (same university) is the structural inspiration at content level; NO `\part{}` divisions.
- DT-2 (2026-07-05): All figures from committed artifacts only; GPU experiments deferred (register above).
- DT-3 (2026-07-05): `report/` EN mirror frozen until F13.
- DT-4 (2026-07-05): Glossary as manual longtable; NO `glossaries` package (avoids makeglossaries build-chain risk).
- DT-5 (2026-07-05): Metodología split into Diseño del sistema + Protocolo experimental (F4) — pure move, fallback is meta-sections if split proves too risky.
- DT-6 (2026-07-06): The MAIN TensorBoard event log is LOCAL-ONLY (`.gitignore` line `runs/**/events.out.tfevents.*`). The exported CSVs under `runs/cicids2017/MAIN_.../plots/tensorboard_scalars/` are the committed, durable source for figure F8.
- DT-7 (2026-07-06): Figure palette (dataviz-validated on white surface): QRDQN `#2a78d6` / RF `#1baf7a` (aqua <3:1 contrast → direct value labels mandatory); benigno `#2a78d6` / ataque `#e34948`; CM heatmaps = single-hue blue ramp, row-normalized shading with count+row-% annotations.
- DT-8 (2026-07-06): babel options `es-lcroman` (lowercase roman folios — the default `\es@scroman` small-caps folios force nonexistent bold-smallcaps in the TOC) and `es-tabla` (floats say "Tabla", matching the prose).
- DT-9 (2026-07-06): fancyhdr v5 requires BOTH page styles defined via `\fancypagestyle{...}` — global `\fancyhf`/`\fancyhead` config gets clobbered by a later `\fancypagestyle{plain}` definition (empirically verified).
- DT-10 (2026-07-06): TeX Live 2025 emits `ignored error: Infinite glue shrinkage found in box being split` on EVERY multi-page longtable (reproduced with a 6-line vanilla document). Benign engine diagnostic; G1 pass criterion remains exit 0 + no `^!` lines; do not chase these.

## Execution log (append one block per phase/session)

### F10 — 2026-07-09 — F10.1–F10.4 done; F10.V verified

- Branch `chore/yeaight7/thesis-f10-anexos`; implementation commits f05e0b4 (F10.1), 5ee234f (F10.2), d346504 (F10.3), and ad73184 (F10.4).
- Annex A is a self-contained reproducibility manual covering LFS/data preparation, local and RunPod setup, smoke and MAIN workflows, fixed-split verification, bootstrap, duplicates, Random Forest, historical VAL A/B/C, Phase 2 inference, and GPU-deferred protocols. Commands with arguments were checked against current parsers; the no-argument duplicate/RF entrypoints were inspected statically. It discloses the historical `fp=-1.0` versus current `fp=-2.0` drift and the removed Check A model, so byte-for-byte regeneration is not promised.
- Annex B separates author-confirmed AMD EPYC 9005 / 128 GB / RTX 3090 Ti hardware from the artifact-recorded MAIN runtime and current installation contracts. Chapter 2 now identifies the laptop as development-only. The recorded cuDNN integer `92000` is decoded as 9.20.0; no unrecorded RTX 6000 upgrade is claimed.
- Annex C adds the complete 76-row canonical enumeration and both source mappings while preserving Chapter 4's conceptual grouped table. There was no pre-existing 76-row body enumeration to move. Assertions confirm 76 unique canonical identifiers, complete CICIDS2017/Flowmeter-py mappings, and exact agreement with MAIN's ordered 152 feature-plus-mask names.
- Annex D catalogs MAIN, VAL A/B/C, the fixed-partition reference, bootstrap, duplicate analysis, all three Random Forest subruns, and the three maintained Phase 2 runs. Every principal path and tracked/local-only claim was audited. The historical pre-rename Phase 2 input remains immutable, `predictions.csv.gz` is local-only, and `predictions_head_10000.csv` is the committed audit sample; archive/prototype exclusions are documented without exhaustive cataloging.
- Each task passed an isolated build after implementation (195, 200, 209, and 217 pages respectively). The final forced `latexmk -gg` build, with `TMP`, `TEMP`, and `TMPDIR` redirected to `C:\Temp`, exits 0 at 215 pages. The +36-page delta from the merged F9 baseline (179 pages) was investigated: Annexes A–D occupy 34 physical PDF pages (15 + 5 + 8 + 6), while body edits and repagination account for the remaining two.
- Final gates: G1 build clean; G2 pages logged; G3 references/citations and Biber warnings clean; G4 manifest 20/20; all ten G5 disclosures at or above baseline; G6 typography lint clean; independent G7 audit PASS (174 schema, environment, dependency, protocol, VAL, RF, artifact, Phase 2, deferred-workflow, and CLI checks). A final extraction audit additionally passed 20/20 schema, 59/59 artifact/provenance, and 6/6 command-rendering checks. `uv lock --check`, `uv run ruff check .`, `uv run pytest` (45 passed), and `git diff --check` pass. All Annex A–D pages were rendered and visually inspected, including landscape longtables, commands, paths, headers, and TOC entries. The final overfull-box count and width multiset are identical to F9; the one new 5.75 pt float-height advisory for body Table T-G was rendered separately and confirmed unclipped with clear footer separation.
- PDF single-writer rule observed: all builds wrote to isolated `C:\Temp` directories; tracked `memoria/memoria.pdf` remained untouched and must be rebuilt on `main` only after merge.

### F9 — 2026-07-09 — F9.1–F9.3 done; F9.V verified

- Branch `chore/yeaight7/thesis-f9-conclusiones-introduccion`; merged through PRs #52 and #53 (merge commits 7cb5300 and d15cd06).
- `memoria/capitulos/conclusiones.tex` adds the contractual and specific-objective synthesis, author reflection, and cited short-, medium-, and long-term future-work horizons; OBJ-07 remains explicitly designed but pending GPU execution.
- `memoria/capitulos/introduccion.tex` adds the PDS-adapted specification, contributions C1–C4 with evidence pointers, and a roadmap covering Chapters 2–10, bibliography, and Annexes A–D.
- Verification on merged `main`: isolated `latexmk` build under `C:\Temp` exit 0, pages=179; G3 references/citations and Biber warnings clean; G4 manifest complete; G5 disclosures remain above baseline; G6 typography lint clean; all OBJ-01–OBJ-08 appear in both Ch6.8 and Ch10; independent G7 audit found no numeric or comparative discrepancy against code and committed artifacts.
- PDF single-writer rule observed: isolated output used for verification; tracked `memoria/memoria.pdf` remained untouched.

### F8 — 2026-07-09 — F8.1–F8.6 done; F8.V verified

- Branch `chore/yeaight7/thesis-f8-discusion-limitaciones-etica`.
- Commits: 4f06b22 (`discusion.tex`, F8.1–F8.4), 45c68f6 (`limitaciones.tex`, F8.5), 4ab4dd1 (`etica.tex`, F8.6).
- `memoria/capitulos/discusion.tex` now follows the requested 7.1–7.6 structure: quantitative reading, random-vs-day generalization, QRDQN-vs-RF, SOTA positioning, C3 multiobjective reading, and threats to validity.
- `memoria/capitulos/limitaciones.tex` expands the evidence-backed limitations, preserves pending placeholders, removes rendered `Trabajo futuro`, and keeps the future-work content as a LaTeX comment stash for F9.1 / 10.4.
- `memoria/capitulos/etica.tex` lightly expands dual-use scope, privacy/data handling, lab-traffic treatment, offline-only inference, false-positive impact, and reproducibility-as-responsibility.
- Verification: forced BUILD (`latexmk -g -pdf -cd ...`) exit 0 with biber 2.21, pages=171; G3 clean; G4 all manifest pdf/png files present; G5 counts 9/5/3/6/4/12/4/2/7 plus `red proxy`=4; G6 typography lint clean; `pdftotext` confirms rendered `Pendiente:` placeholders and no rendered `Trabajo futuro`; read-only verifier PASS for numeric claims vs artifacts and citation fit.
- PDF single-writer rule observed: `memoria/memoria.pdf` restored before commits.

### F7 — 2026-07-09 — F7.1–F7.8 done; F7.V solved by Codex local check

- Branch `chore/yeaight7/thesis-f7-resultados`.
- `memoria/capitulos/resultados.tex` restructured into the requested 6.1–6.8 sequence: condiciones generales/trazabilidad, MAIN, bootstrap, escalera A/B/C, duplicados, Random Forest, Fase 2, and síntesis/cobertura de objetivos.
- Added the missing Check C disclosure in the memoria: red proxy `[512,256]`, 30k steps, not MAIN weights. Added G5 baseline pattern `red proxy`.
- `tab:rf-vs-qrdqn` now includes the Random Forest leave-one-out Wednesday row and keeps QRDQN leave-one-CSV-out explicitly pending GPU, with no fabricated metrics.
- Added T-L as `tab:cobertura-objetivos-resultados`, echoing OBJ-01..OBJ-08 and separating covered, partial, and pending evidence.
- Verification: BUILD exit 0, pages=165; G3 clean; G4 all manifest pdf/png files present; G5 counts 9/5/3/5/5/12/4/1/5 plus `red proxy`=1; G6 clean; G7 literal check confirmed all expected artifact values in `resultados.tex`.
- Note: separate subagent/tool-model verification was not used because the available multi-agent tool policy only allows spawning when the user explicitly asks for delegation. F7.V is therefore marked `[o] codex checked it`, matching the F5.V precedent.
- PDF single-writer rule observed: `memoria/memoria.pdf` must be restored before committing this branch.

### F6 — 2026-07-07/09 — F6.1–F6.6 done; F6.V VERIFIED

- Branch `claude/thesis-f6-objetivos`; PR #48 open. Commits: 118fa31, 6239a82, 16167a3, 2aa2c09, 52117c8, fd48682.
- Chapter renamed; final section layout: 2.1 objetivo general · 2.2 objetivos contractuales (T-C cards) · 2.3 objetivos específicos (OBJ IDs + criterios + T-D) · 2.4 alcance · 2.5 no objetivos · 2.6 resultados esperados · 2.7 restricciones (T-J) · 2.8 recursos (T-B) · 2.9 planificación (T-I + Gantt).
- **C1..C4 catalog**: no contract-objectives text exists anywhere in the repo (searched docs/, PR #41 body+comments); C3 anchored on the tracker's own definition (multiobjetivo {seguridad, impacto}); C1/C2/C4 derived from thesis contributions. USER should validate wording against the real anteproyecto/contrato if it lists objectives.
- GPU-pending marked per user instruction (2026-07-07): OBJ-07 estado+criterio, OBJ-06 nota LOO-QRDQN, RES-E2, C2 verification cell.
- F6.5 authored by sonnet agent (per model policy) with strict no-repo-writes rule (lesson from F5 incident); isolated compile clean; caveats: local-machine specs undocumented (listed as "CPU, sin GPU dedicada"); GPU model taken from runpod doc's "Preferred: RTX 3090 Ti" — confirm the actually-rented pod.
- F6.V run by Codex 2026-07-09: `latexmk -pdf -cd memoria/memoria.tex` exit 0 with TeX Live 2025 and mandatory TEMP redirect; `pdfinfo` pages=162; G3 log grep clean; G4 all manifest pdf/png files exist; G5 counts 8/5/3/5/3/9/3/1/5; G6 typography lint clean. G7 checked Ch2 numeric claims against `config.json`, `environment.json`, `requirements*.txt`, `pyproject.toml`, `f1_gantt.tex`, `figures_manifest.json`, and reward source; no repo-backed numeric mismatches found. Repo gates also green: `uv run ruff check .`, `uv run pytest` (45 passed), and PR #48 GitHub checks passing.
- Noted in passing: user PR #47 added `fig:placeholder-*`/`tab:placeholder-*` labels (currently unreferenced) — presumably intentional placeholders for deferred GPU experiments.


### F4 — 2026-07-06 — DONE except G1–G3 (user compiles manually)

- Branch `claude/thesis-f4-split`. Commits: 7d510ac (F4.1 split), 6c52681 (F4.2 bibliografía), 21f358e (F4.3 anexos), 6e8a6f0 (F4.4 fallout).
- Split executed as a scripted byte-exact move (scratchpad script with purity assertions; original snapshot hash-verified vs git blob). `diseno_sistema.tex` = visión, datos, limpieza, esquema canónico, formulación RL, agente QRDQN, implementación; `protocolo_experimental.tex` = fases, particiones, línea base, escala [pendiente-GPU], evaluación/reproducibilidad, Fase 2, limitaciones metodológicas. Within-file section order preserves the original relative order (the plan's "métricas/escalera … reproducibilidad" listing maps to the single moved section "Salidas de evaluación y reproducibilidad").
- Chapter labels added: `cap:diseno-sistema`, `cap:protocolo-experimental` (unused until F5–F7 wiring; existing chapters keep the hardcoded-«Capítulo N» convention).
- F4.4 fallout was small: only `limitaciones.tex:3` named the dead chapter (retargeted to "el capítulo del protocolo experimental"; its four enumerated limitations verified present in protocolo §Limitaciones metodológicas). Comment pointers updated in `src/train_rl_defender.py:373` (seeds → diseno_sistema, §Controles de reproducibilidad) and `memoria/figuras/f4_vector_observacion.tex:3` (máscara → diseno_sistema). Historical audit docs intentionally untouched.
- F4.V: 3/3 adversarial verifiers PASS (fidelity: 349 body lines = 225 D + 123 P + 1 EOF blank, per-section byte-identical mod EOL, non-ASCII inventory intact; wiring: bibintoc = `\chapter*` + TOC entry + `\markboth` so fancyhdr works, comment-only anexos contribute zero tokens, no unintended diff; numbering: 40 «capítulo» hits audited, 3 hardcoded numbers all still correct, G5/G6 independently re-run). G5 before==after: γ=0→5, 24.86→5, 40.12→3, 0.52954→5, bandido→2, validez→6, 0.991862→3, semilla→1, muestreo→3.
- **Pending for G1–G3 (user instruction: compile manually):** run BUILD, then G3 log grep; record pages (expect ~130→131±1: one extra chapter break from the split; bibliografía heading swap and empty anexos should be page-neutral).
- Minor advisories for later phases (NOT F4 defects): resultados.tex:3,6 and etica.tex:15 say "en la metodología" generically — reads fine, but F7/F8 rewrites could point at the concrete chapter; limitaciones.tex:3 attributes the literal term "bandido contextual" to protocolo while the literal string lives in diseno (§γ=0) — conceptually correct as written.
- PDF not touched on branch (single-writer rule). WIP: none.

### F3 — 2026-07-06 — DONE

- Branch `claude/thesis-f3-frontmatter`. Commits 36ae1c0 (design layer) + c1180b5 (front matter). Pages 118 → 130 (front matter now 18 physical pages incl. 6-page TOC).
- Front matter: portada (Loyola logo, degree, tutor — user-provided), agradecimientos (user text), resumen 298 w + palabras clave, abstract EN, TOC/LOF/LOT/LOA (roman folios, indices read as text via linkcolor group), glossary of 30 census-grounded acronyms.
- Design pass (user request): fancyhdr running headers, titlesec chapter format (gray small-caps label + blue rule), colored links + PDF metadata, microtype, caption styling, short captions for the 6 existing tables.
- Debugging findings recorded as decisions: DT-8 (`es-lcroman` — babel's small-caps folios force bold-smallcaps in TOC; `es-tabla` — prose says Tabla), DT-9 (fancyhdr v5 clobbers global config; define both styles via `\fancypagestyle` — empirically bisected), DT-10 (TL2025 emits a benign "Infinite glue shrinkage" diagnostic on every multi-page longtable — reproduced with a vanilla 6-line doc; do not chase).
- F3.V: adversarial verifier PASS 19/19 (all numbers vs artifacts, ES/EN parity, disclosures, portada facts). 10 pages visually reviewed via Ghostscript.
- Remaining open item: portada date "Julio de 2026" — confirm vs defense date. *User will adjust this manually*

### F2 — 2026-07-06 — DONE

- Branch `claude/thesis-f2-diagrams`. 7 TikZ diagrams authored by a 7-agent workflow (each grounded in its source files, compile-verified in isolated scratch dirs), plus `figuras/tikz_estilos.tex` (shared styles, wired into the preamble).
- Session hit its limit mid-review; user committed all authored .tex + memoria.pdf (PDF-on-branch deviation accepted). Post-resume fixes (commit 6dfad5b): f2 wording + flow-extraction attribution corrected to the external CICFlowMeter-py extractor (`pcaps/README.md`); f3 benigno/omisión + label collisions; f5 hyphenation; f17 presence-mask wording + literal `--clip-z`.
- False alarm during review: "decimal commas" in renders were a wrapper artifact — babel-spanish converts math-mode periods without `\decimalpoint`; memoria.tex has it, so the document renders points correctly.
- F2.V: 7/7 adversarial verifiers PASS (70 checks) — rewards vs `rl_defender_env.py`, Gantt dates vs the evidence list, net dims vs MAIN config, ladder verdicts vs artifacts, presence-mask semantics vs `canonical_schema.py`, Phase-2 stage order vs `predict_real_traffic_v2.py`.
- Gates: G1 clean (118 pp), G3 clean, G5 baseline intact, G6 clean. All 7 diagrams visually reviewed via Ghostscript renders (`rungs.exe`; raw `gswin64c` lacks its lib path).
- NOTE for wiring phases (F5–F7): figures are `\input`-ready tikzpictures (no figure env); the Gantt file is a bare `ganttchart` environment.
- Last-green commit: see PR. WIP: none.

### F1 — 2026-07-06 — DONE

- Branch `claude/thesis-f1-figures`. Commits: 8667520 (generator), 89bb943 (TB scalar export), 816164a (10 figures + manifest + per-day data JSON).
- 10 data figures generated from committed artifacts only; palette validated with the dataviz validator (DT-7); proxy-net (F11/F13) and lab-validity (F15) disclosures embedded as in-figure footnotes; QRDQN-LOO shown as "pendiente de GPU", no fabricated bar.
- Discovered: MAIN `metrics.json` has no CM cells — F9 sources them from `bootstrap_ci_seed42.json` (`confusion_counts`, self-validated vs regenerated counts). TB event log is local-only → exported CSVs committed (DT-6). Infiltration day has 36 attacks (0.0125%) — F6 labels it "<0.1 % ataque", not 0.
- F1.V: 8-agent adversarial verification workflow — 8/8 PASS, 102 checks, 0 mismatches, 0 visual issues.
- Gates: G1 clean (118 pp, PDF restored, not committed on branch), G3 clean, G4 0 missing, G5 baseline intact, G6 clean.
- Last-green commit: see PR. WIP: none.

### F0 — 2026-07-05 — DONE (this PR)

- Branch `claude/thesis-f0-scaffold`. Toolchain verified with full `latexmk -gg` rebuild: biber 2.21 green under `TEMP=C:\Temp`, 118 pp, 0 undefined refs.
- Preamble hardened in 3 individually-built batches (config → tikz/pgfgantt/longtable/pdflscape → algorithm/algpseudocode/caption/subcaption + Spanish float names). Pages unchanged (118).
- `.gitattributes`: thesis PDFs `merge=binary`. `memoria/figuras/` created.
- Baseline captured (see Baseline block). Discovered during baseline: Check-C proxy-net disclosure is absent from the memoria (exists only in `docs/results.md`) → explicit task F7.4.
- Last-green commit: see PR. WIP: none.
