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
6. Resume at the first open task, on that phase's branch (`claude/thesis-f<N>-<slug>`).
   Never start a new phase with a dirty tree. Never edit `memoria.tex`/`memoria.bib` in parallel branches.

## Environment (verified 2026-07-05)

- TeX Live 2025 at `D:\texlive\2025`. All needed packages present (tikz, pgfgantt, algorithm,
  algpseudocode, longtable, pdflscape, caption, subcaption — verified with kpsewhich).
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

`[ ]` todo · `[~]` in progress (WIP note required in Execution log) · `[x]` done & verified · `[-]` won't-do / deferred · `[?]` blocked on user decision

## Verification gates

| Gate | What | Command / rule |
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
| `\\gamma = 0` | 5 | γ=0 contextual-bandit |
| `24\.86` | 3 | test-in-train duplicate leakage % |
| `40\.12` | 3 | test-attack duplicate leakage % |
| `0\.52954` | 5 | Check C day-split attack recall |
| `bandido contextual` | 1 | bandit framing |
| `validez externa limitada` | 5 | Phase-2 external validity |
| `0\.991862` | 3 | Phase-2 accuracy (lab-only context) |
| `una sola semilla` | 1 | single-seed limitation |
| `precisi.n de muestreo` | 2 | bootstrap = sampling precision, not seed variance |

- **Known-missing disclosure (to ADD in F7, then append here):** Check C used a proxy network
  (`[512,256]`, 30k steps), NOT the MAIN weights — currently only in `docs/results.md`, absent from the memoria.
- Chapter word counts at baseline: intro 2,139 · objetivos 2,964 · estado del arte 8,456 · metodología 8,488 · resultados 1,321 · discusión 632 · limitaciones 734 · ética 665.

## Honesty invariants (non-negotiable during ALL edits)

γ=0 contextual-bandit disclosure · duplicate-leakage 22.30/24.86/40.12% cited · Phase-2 framed
"laboratorio doméstico cerrado, generado por el operador, validez externa limitada" · single-seed
limitation · Check C = proxy net, no MAIN weights (from F7 on) · bootstrap CI = precisión de
muestreo, no varianza de semilla · RF beats QRDQN in-distribution (0.99676 vs 0.98445 F1) stated
plainly · pending GPU experiments always "diseñados e implementados, ejecución pendiente", never
with numbers.

## Git strategy

- One branch per phase `claude/thesis-f<N>-<slug>` → PR to `main`. No force-push, no history rewrite (D-6).
- One atomic commit per task ID; tree build-green at every commit; `wip(F5.3): ... [DO NOT MERGE]` allowed at session end with a WIP note here.
- **PDF single-writer rule**: phase branches NEVER commit `memoria/memoria.pdf` (restore with
  `git checkout -- memoria/memoria.pdf` before committing). Rebuild + commit the PDF only on `main`
  right after each merge: `build: rebuild memoria.pdf (Fn, NNN pp)`. `.gitattributes` marks it `merge=binary`.
- Commit style: `type: summary (F1.2,F1.3)`. **No AI co-author trailers, no "Generated with" footers** in commits or PR bodies (user rule, 2026-07-06).

## Subagent / model policy

| Work | Model |
|---|---|
| Mechanical LaTeX edits, wiring, tracker upkeep | haiku/sonnet |
| Figure scripts, TikZ, TB export | sonnet |
| Spanish academic prose (resumen, results, discusión, conclusiones, style pass) | top model |
| Adversarial verification (G5/G7, citation context, EN fidelity) — separate agent from the writer | top model |

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
| F5.V | G7 numbers check (top model) + gates G1–G3, G5, G6 | TOP | F5.* | [-] codex checked it | |

## F6 — Ch2: Objetivos, alcance y planificación

| ID | Task | Model | Dep | Status | Evidence |
|---|---|---|---|---|---|
| F6.1 | Rename chapter; OBJ-C1..C4 contract catalog (T-C spec tables: ID/Descripción/Tipo/Verificación/Capítulos) | TOP | F4 | [ ] | |
| F6.2 | OBJ-01..08 IDs + criterio de verificación on existing 8 subsections | TOP | F6.1 | [ ] | |
| F6.3 | T-D matriz de trazabilidad (objetivos ↔ secciones ↔ RUN_IDs ↔ estado; OBJ-07 = Pendiente-GPU) | TOP | F6.2 | [ ] | |
| F6.4 | 2.7 Restricciones (T-J RES-xx: factores dato / estratégicos) | TOP | F6.1 | [ ] | |
| F6.5 | 2.8 Recursos (T-B HW/SW from `requirements*.txt`, `docs/runpod_main_experiment.md`) | sonnet | F6.1 | [ ] | |
| F6.6 | 2.9 Planificación temporal: T-I hitos + embed Gantt F1 | TOP | F2.7 | [ ] | |
| F6.V | Gates + G7 | TOP | F6.* | [ ] | |

## F7 — Ch6 Resultados expansion (heaviest content phase)

| ID | Task | Model | Dep | Status | Evidence |
|---|---|---|---|---|---|
| F7.1 | 6.1 Condiciones generales + T-G + traceability statement | TOP | F1, F5 | [ ] | |
| F7.2 | 6.2 Experimento 1 MAIN: Condiciones / Dinámica (embed F8) / Resultados (embed F9) / Discusión breve | TOP | F7.1 | [ ] | |
| F7.3 | 6.3 bootstrap (embed F10; keep "precisión de muestreo" verbatim) | TOP | F7.1 | [ ] | |
| F7.4 | 6.4 escalera A/B/C (embed F11) + **ADD proxy-net disclosure** (`[512,256]`, 30k steps, no MAIN weights) — then append pattern to G5 baseline | TOP | F7.1 | [ ] | |
| F7.5 | 6.5 duplicados/fuga (embed F12; numbers verbatim) | TOP | F7.1 | [ ] | |
| F7.6 | 6.6 RF baseline (embed F13+F14; add LOO row to tab:rf-vs-qrdqn; RF in-distribution win stated plainly) | TOP | F7.1 | [ ] | |
| F7.7 | 6.7 Fase 2 (embed F15; lab-validity framing verbatim) | TOP | F7.1 | [ ] | |
| F7.8 | 6.8 síntesis + cobertura de objetivos (T-L; echo OBJ IDs) | TOP | F7.2–7.7, F6 | [ ] | |
| F7.V | G7 adversarial numbers check (separate top-model agent) + all gates | TOP | F7.* | [ ] | |

## F8 — Ch7 Discusión + Ch8 Limitaciones + Ch9 Ética

| ID | Task | Model | Dep | Status | Evidence |
|---|---|---|---|---|---|
| F8.1 | 7.1–7.3 expand quantitatively (exact table values) | TOP | F7 | [ ] | |
| F8.2 | 7.4 posicionamiento vs SOTA (leakage caveat on literature numbers; cite Layeghy2022, Boukhamla2021, Cantone2024) | TOP | F7 | [ ] | |
| F8.3 | 7.5 lectura multiobjetivo {seguridad, impacto} (contract C3) | TOP | F7 | [ ] | |
| F8.4 | 7.6 amenazas a la validez (short, cross-ref Ch8) | TOP | F7 | [ ] | |
| F8.5 | Ch8: expand each limitation 1–2 paragraphs; MOVE "Trabajo futuro" content out (staged for 10.4) | TOP | F7 | [ ] | |
| F8.6 | Ch9 Ética light expansion (privacidad + lab-traffic handling; no new claims) | TOP | — | [ ] | |
| F8.V | Gates + G5 strict (this phase touches the disclosure-dense chapters) | TOP | F8.* | [ ] | |

## F9 — Ch10 Conclusiones (new) + Ch1 expansion

| ID | Task | Model | Dep | Status | Evidence |
|---|---|---|---|---|---|
| F9.1 | New `capitulos/conclusiones.tex`: 10.1 objetivos contractuales echo · 10.2 objetivos específicos echo (OBJ-07 "diseño completado, ejecución pendiente") · 10.3 reflexión del autor · 10.4 líneas futuras corto/medio/largo plazo (each with cite; Zhang2025OpenSet here) | TOP | F6, F8 | [ ] | |
| F9.2 | Ch1: PDS-adapted spec (1.2.2), contributions C1–C4 with evidence pointers, lead-in | TOP | F6 | [ ] | |
| F9.3 | Ch1 §1.6 roadmap rewrite naming ALL chapters + anexos | TOP | F9.1 | [ ] | |
| F9.V | Gates; every OBJ ID from Ch2 appears in Ch6.8 or Ch10 | TOP | F9.* | [ ] | |

## F10 — Anexos A–D

| ID | Task | Model | Dep | Status | Evidence |
|---|---|---|---|---|---|
| F10.1 | Anexo A Manual de reproducibilidad (from `docs/reproducibility.md` + exact run commands) | sonnet | F4 | [ ] | |
| F10.2 | Anexo B Entorno HW/SW (versions from `requirements*.txt`, `environment.json`) | sonnet | F4 | [ ] | |
| F10.3 | Anexo C Esquema canónico completo (MOVE 76-feature enumeration out of Ch4 body) | sonnet | F4 | [ ] | |
| F10.4 | Anexo D Catálogo de artefactos por ejecución | sonnet | F4 | [ ] | |
| F10.V | Gates | cheap | F10.* | [ ] | |

## F11 — Estado del arte trim + bibliography weave

| ID | Task | Model | Dep | Status | Evidence |
|---|---|---|---|---|---|
| F11.1 | Tighten 8–10% in the 3 longest sections (datasets, supervised DL, riesgos metodológicos) — nothing deleted or moved out | TOP | F9 | [ ] | |
| F11.2 | T-H related-work RL-NIDS table weaving NIDSRL2023, RLTechniques2023NIDS, Sanusi2023DRLIDS, Umer2022RLRLIDS, Cevallos2023DRLIDSBP, DDPG2025AttackDetection, HCLRIDS2025IoMT, DRLIDSSDN2025 | TOP | F11.1 | [ ] | |
| F11.3 | Weave remaining uncited: Layeghy2022, Boukhamla2021, Cantone2024, DatasetSurvey2025, Ozgur2016, Rodriguez2022, TrainingData2025, Farrukh2022, Pekar2024 (one claim-bearing citation each); prune Oyelakin2023Overview if no honest slot | TOP | F11.1 | [ ] | |
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
| G.1 | leave-one-CSV-out (QRDQN) | `uv run python -m src.validate_leave_one_csv_out` (see module docstring for args) | [-] pendiente de GPU |
| G.2 | Training-size ladder (100k/250k/500k/1M/2M, nested-prefix) | `uv run python -m src.train_rl_defender --preset full --train-max-rows <N> ...` per `experiments/cicids2017_qrdqn_experiments.md` | [-] pendiente de GPU |
| G.3 | Multi-seed variance study (MAIN profile, ≥3 seeds) | same as MAIN with `--seed {43,44,45}` | [-] pendiente de GPU |

## Open items needing USER input

| Item | Needed for | Status |
|---|---|---|
| ~~Exact degree + faculty line for portada~~ | F3.1 | resolved 2026-07-06: Grado en Ingeniería Informática y Tecnologías Virtuales, Universidad Loyola |
| ~~Tutor name confirmation~~ | F3.1 | resolved 2026-07-06: Alfonso Carlos Martínez Estudillo |
| ~~Agradecimientos text~~ | F3.3 | resolved 2026-07-06 (Juan y Esmeralda, Carlos, Alfonso Carlos) |
| ~~Optional: phase labels for Gantt bands~~ | F2.7 | resolved — Gantt built from contract dates + repo evidence (first commit, run timestamps) |
| Portada date currently "Julio de 2026" | F3.1 | [?] confirm or adjust to defense date - User will adjust manually |

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
