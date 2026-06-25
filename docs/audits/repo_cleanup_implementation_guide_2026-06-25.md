# Guía de implementación — limpieza y realineamiento del repo TFG_CYBER_AI

**Fecha:** 2026-06-25 · **Rev. 3** (consolida y reemplaza los audits previos; reencuadre del marco de runs/hiperparámetros).
**Tipo:** auditoría read-only + plan por fases (checklist). **Único audit vivo** del repo: consolida `tfg_cyber_ai_audit.md` y `stale_claims_diagnosis_2026-06-15.md` (retirados 2026-06-25; en historia git). Ver §11.
**Estado:** reorg de `docs/` y limpieza de Personal Research ya ejecutadas/commiteadas. El resto del plan (Fases 1–6) sigue pendiente.
**Alcance auditado:** docs `.md`/`.tex` (grep), código `src/`/`scripts/`/`tests/`, `.gitignore` + tracking, JSON pequeños citados en `docs/results.md`. No se leyeron PDFs, datasets, PCAPs ni recorridos completos de `runs/`.

## 0. Cómo usar esta guía

Trabaja por fases (§7). Cada item tiene casilla `[ ]`. Verifica la evidencia citada antes de tocar nada. Tras cada fase corre los comandos de §8. Respeta "No tocar todavía" (§9).

---

## 1. Marco oficial del proyecto (decidido por el autor, 2026-06-25)

**Tronco oficial / experimentos oficiales:**
- **MAIN** = `MAIN_qrdqn_cicids2017_canonical_full_random_20260609_193655` (preset `full`, 3.000.000 timesteps, train 2.264.594 / test 566.149). Es **el** resultado principal.
- **Experimentos secundarios** = runs casi idénticos a MAIN (mismos hiperparámetros, **mismo split de test fijo**) pero con **menos filas de training** (`--train-max-rows`), para validar si compensa entrenar con más/menos datos. **Aún no existen** como artefactos comprometidos (la tabla "Training-Size Benchmark" en `docs/results.md` está "pending").

**Probes pre-diseño (NO forman parte del tronco):** runs/modelos que el autor lanzó *antes* de fijar el diseño experimental, solo para ver si funcionaba. No son errores ni confusiones: eran exploración. En comparación con MAIN, hoy son técnicamente smoke/test:
- `C01` smoke + full, `C02`, `C03` (full) y `C03_..._fast_day_..._0` (huérfano: solo `events.out.tfevents`).
- `A01__arch256x256...`, `A02_dqn_arch512x256...`, `rl_defender_dqn.zip`.
- **3 runs `MAIN_*fast_random` del 09-jun** (`...185427`, `...190202`, `...191901`): llevan prefijo MAIN pero son `fast` → smoke previos al MAIN full real. El prefijo "MAIN" en ellos es engañoso.

**Hiperparámetros oficiales = los actuales del perfil `main-experiment` (usados en MAIN).** Los antiguos eran exploración y no se consideran canónicos. El perfil `default` del script queda como conveniencia dev/smoke, **no** oficial.

### 1.1 Verdad canónica de hiperparámetros (fuente: `src/train_rl_defender.py` + `MAIN/config.json`)

| Parámetro | `main-experiment` (OFICIAL, run MAIN) | `default` (dev/smoke) |
|---|---|---|
| `REWARD_CONFIG` | tp 1.5 / fp **-2.0** / fn -5.0 / om 0.0 | igual |
| `gamma` | **0.0** | **0.0** |
| `net_arch` | **[1024, 1024, 512]** | [512, 256] |
| `n_quantiles` | 200 (explícito) | 200 (explícito) |
| `exploration_fraction` | **0.10** | 0.005 |
| `exploration_final_eps` | 0.02 | 0.01 |
| `learning_rate` | 5e-5 | 1e-4 |
| `batch_size` | 2048 | 512 fast / 2048 full |
| `timesteps` | 3.000.000 | 25k fast / 100k full |

CI (`.github/workflows/ci.yml`): `uv sync --all-extras`, check esquema `==76`, `pytest tests/`, `ruff check .`. **Ningún script de `src/`/`scripts/` está muerto** — todos tienen referencias verificadas.

---

## 2. Decisiones tomadas (2026-06-25)

1. **Probes pre-diseño → archivar (siguen tracked):** mover a `runs/archive/` y `models/archive/` con README de nota. Conservan trazabilidad fuera del tronco.
2. **C0x en tablas de resultados → sacar de las tablas oficiales:** las tablas oficiales muestran solo MAIN + secundarios; C0x (y artefactos claramente históricos) van a un apéndice/histórico.
3. **Binarios pesados → dejar de trackear los no-oficiales:** `git rm --cached` + `.gitignore` para `events.out.tfevents.*`, `runs/**/model.zip` (duplicados) y binarios de runs futuros no-oficiales. **Sin reescribir historia.**

### Reconciliación 1 ↔ 3 (importante al ejecutar)
- Los **probes archivados** conservan tracked su `.zip` + JSON pequeños (excepción explícita de la decisión 1).
- La decisión 3 (destrackear) aplica a lo **redundante/pesado**: todos los `events.out.tfevents.*` (incluidos los dirs `_0`), los `runs/**/model.zip` (duplican `models/`), y futuros binarios no-oficiales.
- **Queda tracked como oficial:** `models/MAIN_qrdqn_cicids2017_canonical_full_random_20260609_193655.zip` + los JSON pequeños del run MAIN (config/metrics/scaler/percentiles/feature_names/environment/manifest) + los futuros secundarios.

---

## 3. Hallazgos priorizados

### P0 — claims peligrosos para la defensa

**D1 · Hiperparámetros antiguos presentados como actuales en docs de defensa TRACKED.**
`docs/Personal Research/qrdqn-research-report.{md,tex,pdf}`, `models-parameters-and-validation-thesis-defense-report.*`, `deep-defense-research/03-...*` describen `gamma=0.99`, `net_arch=[512,256]` como arquitectura principal y `n_quantiles`/exploración como "implícitos". Realidad oficial (§1.1): `gamma=0.0`, `[1024,1024,512]`, `n_quantiles=200` explícito, `exploration_fraction=0.10`. No eran errores en su día (exploración), pero hoy esos docs **presentan valores no-oficiales como el config principal**. **Riesgo:** el tribunal lee un PDF que no cuadra con el run MAIN. La diagnosis del 15-jun solo cubrió los `.md` (ignorados); aquí lo crítico son los `.tex/.pdf` versionados.

**D2 · `graphify-out/` stale con claim objetivamente falso.**
Generado en `c97b734c` (anterior a `memoria/`). Contiene `Reward Config Historical (fp=-2.0) vs Current (fp=-1.5) Discrepancy.md` (fp=-1.5 nunca fue actual) y nodos `net_arch=512,256 (main training)`, `exploration_fraction=0.005`, `n_quantiles (implicit default)`. **Riesgo agravado:** `AGENTS.md` ordena a los agentes empezar por el grafo. → regenerar (`graphify .`) o marcar stale + suavizar `AGENTS.md`.

### P1 — reproducibilidad / trackability

**R1 · `test_set_sha256` aspiracional presentado como hecho.** `.github/AGENT_CONTEXT.md` dice "every load records `test_set_sha256`…", pero `MAIN/config.json → split_metadata` solo tiene conteos/ratios, sin hash; el manifiesto de referencia está "pending mint on RunPod". → marcar como mecanismo futuro.

**R3 (NUEVO) · Rutas de Phase 2 rotas tras el rename `_MAIN`.** El dir se renombró a `runs/phase2/P2v2_pred_20260610_161231_MAIN/` (commit 61770a6), pero docs mantenidos siguen apuntando a la ruta vieja sin `_MAIN`:
- `docs/results.md` (≈204, 251), `docs/AGENT_CONTEXT.md` (≈54), `docs/phase2_plan.md` (≈119, 134), `experiments/cicids2017_qrdqn_experiments.md` (≈159).
→ actualizar todas a `..._161231_MAIN`.

**H1 · Inversión de trackability (Personal Research) — RESUELTO 2026-06-25.** Antes `.gitignore` ignoraba los `.md` (fuente) mientras versionaba `.tex`+`.pdf` (derivados). Ahora se versiona **solo el `.md`**; `.tex`/`.pdf` eliminados; enlace de `docs/README.md` apuntado al `.md`. Notas marcadas como investigación, no fuente de verdad.

**R2 · Comparación RF "Check C" con splits distintos.** `baseline_random_forest_comparison/results_rf.txt` Sweep-2 (day split) testea solo Viernes (3 CSV); el Check C de QRDQN testea Jueves+Viernes (n_test 1.162.213). `docs/results.md` invita a compararlos directamente. → aclarar o realinear. Nota: el RF **random split** (test 566.149 = idéntico al de MAIN) sí es baseline oficial alineado con MAIN.

### P2 — documentación stale / duplicada / mal ubicada

- **D3 · README docmap → tesis equivocada.** `README.md` lista `report/report.tex` como "thesis report source draft"; la canónica ahora es `memoria/`. `docs/README.md` no lista `memoria/`. → corregir docs.
- **D4 · Tablas oficiales de `docs/results.md`.** Hoy la tabla "CICIDS2017 Training Runs" lista C01/C02/C03 y omite MAIN en "Artifact Locations > Training". Por decisión 2: dejar tablas oficiales = **MAIN + secundarios**; mover C0x a apéndice histórico; añadir el dir de artefactos de MAIN.
- **D5 · Audits previos consolidados y retirados (HECHO 2026-06-25).** `tfg_cyber_ai_audit.md` y `stale_claims_diagnosis_2026-06-15.md` se revisaron, sus ítems útiles se volcaron aquí (§11) y se eliminaron (historia en git). Esta guía es el **único audit vivo**.
- **C1 · Optuna desincronizado.** `optuna==4.9.0` en `requirements.txt` pero **no en `pyproject.toml`**; `src/tune_hparams.py` lo importa y CI usa `uv sync`. → añadir a `pyproject.toml` o marcar `tune_hparams` opcional.
- **C2 · README sin descarga/SHA de CICIDS2017** (de `tfg_cyber_ai_audit`). `README.md` no da link de descarga ni hash del release → riesgo de usar una variante distinta del dataset. → añadir URL + SHA-256 (o nota de procedencia).
- **D6 · DEFENSA: Phase 2 solo benign-only + MAIN ausente** (de `stale_claims` 3.1, ampliado). `DEFENSA_TFG_PROGRESO.md` (≈271/274) narra Phase 2 solo con runs benign-only (`block_rate 1.0/0.0`); no cita el run etiquetado `..._161231_MAIN`. Los `DEFENSA_*` centran C03 ("mejor run histórico") y **no mencionan MAIN**. → añadir el run etiquetado y centrar MAIN; C0x = probes pre-diseño.
- **D7 · (baja) terminología Check A/B/C** (de `tfg_cyber_ai_audit`). El texto de methodology no mapea a los nombres `Check A/B/C` del código. `report/` aparcado → baja prioridad; aplicar a `memoria/` si interesa.
- **R4 · (menor) z-score post-clipping** (de `stale_claims` 5.2). En `predict_real_traffic_v2.py`, `compute_diagnostics()` corre tras `apply_z_clipping()` con `--clip-z`; los `z_abs_*` describen el estado post-clipping, no el drift bruto. → matizar en docs Phase 2 si se documenta el drift.

### P2/P3 — higiene de artefactos tracked
- **H2 · Binarios/generados versionados sin regla global** (ver decisión 3): `runs/**/events.out.tfevents.*`, dirs `*_0/` (solo TB), `runs/**/model.zip` (duplican `models/`). → destrackear + `.gitignore`.
- **H3 · `pcaps/deprecated_*` tracked** (5 pcap/csv, sin refs de código). → archivar/borrar.
- **Phase 2 probes antiguos** (`P2_pred_*` v1, `P2v2_*` de feb–abr): exploratorios. Dos están citados como evidencia de domain-shift (`P2v2_pred_20260224_004121` block_rate 1.0; `P2v2_pred_20260408_230318` block_rate 0.0) — conservarlos (reencuadrados) o archivarlos con nota; el resto → archivar.

---

## 4. Buckets de acción

### Archivar (mover a `archive/`, siguen tracked) — decisión 1
| Origen | Destino |
|---|---|
| `runs/cicids2017/{C01_smoke,C01_full,C02,C03_full,C03_fast_day_*_0}` | `runs/archive/cicids2017/` |
| `runs/cicids2017/MAIN_*fast_random_{185427,190202,191901}(+_0)` | `runs/archive/cicids2017/` |
| `models/{C01_*,C02_*,C03_*,A01_*,A02_*,rl_defender_dqn.zip}` | `models/archive/` |
| `models/{MAIN_*fast_*185427,190202,191901}.zip` | `models/archive/` |
| Phase 2 probes feb–abr (salvo los 2 citados) | `runs/archive/phase2/` |
- Añadir `archive/README.md`: "runs/modelos exploratorios previos al diseño experimental; no oficiales".

### Destrackear (git rm --cached + .gitignore, sin borrar de disco) — decisión 3
- `runs/**/events.out.tfevents.*` (todos, incl. dirs `_0`).
- `runs/**/model.zip` (duplican `models/`).
- Regla `.gitignore` para futuros binarios de runs no-oficiales.

### Corregir documentación
D1, D2 (decisión), D3, D4, D5, R1, R2, R3, C1.

### Borrar (opcional, con cautela)
| ID | Qué | Verificar no-uso | Riesgo | Alternativa |
|---|---|---|---|---|
| B1 | `pcaps/deprecated_*` | `rg -n "deprecated_lab" --glob '!graphify-out'` | Bajo | Mover a `archive/` |
| B2 | `graphify-out/` | output de `graphify .` | Medio (`AGENTS.md` lo usa) | Regenerar en vez de borrar |

### Dejar intacto
`src/` (sin código muerto), `tests/`, run+modelo MAIN, `datasets/CICIDS2017/*.csv`, `datasets/nsl_kdd/` (histórico etiquetado), `memoria/`, `GEMINI.md`/`CLAUDE.md`/`.remember/`, `scripts/export_tensorboard_scalars.py`, `scripts/verify_fixed_test_split.py`, `src/load_nsl_kdd.py` + `models/rf_nslkdd.joblib` (histórico, referenciado).

---

## 5. Naming going-forward (recomendación)
- Reservar el prefijo **`MAIN`** solo para el run oficial full (193655) y el patrón de los **secundarios** (mismo diseño, menos training). Por eso los 3 `MAIN_*fast` se archivan.
- Secundarios sugeridos: `MAIN_qrdqn_cicids2017_canonical_full_random_t{N}_<ts>` (con `--train-max-rows N`, mismo seed/test fijo).

---

## 6. Puntos de decisión

**6.1 — Documentos tipo-tesis (RESUELTO 2026-06-25):**
- `memoria/` = tesis **canónica oficial** (ES).
- `report/` (EN) = canónica en inglés, **aparcada** por ahora (se acepta que quede algo desactualizada en secciones nuevas). Se conserva en sitio.
- `docs/informe.tex` / `.pdf` = borrador inicial **obsoleto** de la memoria → **archivado** en `docs/archive/` (ver §10).

**6.2 — Tracking de `docs/Personal Research/` (RESUELTO 2026-06-25):** son notas de investigación (Deep Research), **no fuentes de verdad** — los valores que mencionen (p.ej. un `gamma`) son ilustrativos, no canónicos. Se conserva **solo el `.md`** (formato fuente único); se eliminaron los `.tex` y `.pdf` derivados y se quitó el ignore de los `.md`. No se pierde contenido: los `.md` eran superconjunto (incluían `07-artefactos-...` que no tenía `.tex`).

**6.3 — Framing prospectivo de la introducción (DECISIÓN DELIBERADA — no re-flaggear):** el `tfg_cyber_ai_audit` marcó como "overclaiming" que la introducción presente la comparación empírica QRDQN vs RF como contribución. **No es overclaiming: es intencional** — los resultados se completarán y la redacción se adelanta a ese punto. El texto vigente en `report/` y `memoria/` (`introduccion.tex:29`) ya está en tono neutro/prospectivo ("designed to produce honest evidence… without overclaiming"). **No suavizar más** (sería *underclaiming*).
> **Checkpoint futuro (al cerrar resultados):** verificar que la introducción y la sección de resultados de `memoria/` **y** `report/` reflejen los resultados ya logrados. El riesgo real es olvidar endurecer el texto cuando toque, no el overclaim.

---

## 7. Plan por fases (checklist)

### Fase 1 — Claims peligrosos (P0)
- [ ] **D1**: corregir hiperparámetros en los docs de defensa. Texto canónico a insertar:
  > Config oficial (run MAIN, perfil `main-experiment`): `gamma=0.0`, `net_arch=[1024,1024,512]`, `n_quantiles=200` (explícito), `exploration_fraction=0.10`, `lr=5e-5`, `timesteps=3_000_000`. Los valores `[512,256]`, `gamma=0.99` o `exploration_fraction=0.005` fueron exploración previa / perfil `default`, no el experimento oficial.
- [x] **D2**: `graphify .` para regenerar desde `HEAD`, **o** marcar `GRAPH_REPORT.md` stale + suavizar la regla "empezar por el grafo" en `AGENTS.md`. **DONE WITH GEMINI CLI**

### Fase 2 — Documentación obsoleta (P1/P2)
- [ ] **R3**: actualizar todas las rutas a `runs/phase2/P2v2_pred_20260610_161231_MAIN/` (results.md, AGENT_CONTEXT.md, phase2_plan.md, experiments/...).
- [ ] **R1**: marcar `test_set_sha256` como mecanismo previsto (pendiente de mint); aclarar que MAIN no lo incluye.
- [ ] **R2**: aclarar en `docs/results.md` que el day-split RF (Viernes) ≠ Check C QRDQN (Thu+Fri); mantener RF random-split como baseline oficial de MAIN.
- [ ] **D3**: `README.md` docmap → `memoria/` (marcar `report/` histórico).
- [ ] **D4**: reestructurar `docs/results.md` → tablas oficiales = MAIN + secundarios; C0x a apéndice histórico; añadir dir de artefactos de MAIN.
- [ ] **C1**: añadir `optuna==4.9.0` a `pyproject.toml` (o marcar `tune_hparams` opcional).
- [ ] **C2**: añadir URL de descarga + SHA-256 (o nota de procedencia) de CICIDS2017 a `README.md`.
- [ ] **D6**: en `DEFENSA_*`, añadir el run Phase 2 etiquetado `..._161231_MAIN` y centrar MAIN como run oficial (C0x = probes).

### Fase 3 — Archivado de probes (decisión 1)
- [ ] Crear `runs/archive/` y `models/archive/` + `archive/README.md`.
- [ ] `git mv` de los probes según la tabla de §4.
- [ ] Reencuadrar `experiments/cicids2017_qrdqn_experiments.md`: C0x = probes pre-diseño (no "best committed").

### Fase 4 — Código
- [x] **D5**: audits consolidados en esta guía; `tfg_cyber_ai_audit.md` y `stale_claims_diagnosis_2026-06-15.md` retirados (HECHO).
- [ ] Confirmar con `rg` (§8) que no hay código muerto. Reubicar `pcaps/deprecated_*` (B1) y `scratch/`.
- [ ] **D7** (baja): mapear terminología `Check A/B/C` en methodology si se aplica a `memoria/`.

### Fase 5 — Tracking de binarios (decisión 3)
- [ ] `.gitignore`: añadir `runs/**/events.out.tfevents.*`, `runs/**/model.zip`, `scratch/`.
- [ ] `git rm --cached` de esos artefactos ya commiteados (no borrar de disco; **no** `filter-repo`/BFG).
- [ ] Mantener tracked: `models/MAIN_*full_*193655.zip`, JSON pequeños del run MAIN, probes archivados (`.zip` + JSON), futuros secundarios.
- [ ] **H3**: borrar/archivar `pcaps/deprecated_*`.

### Fase 6 — Validación final
- [ ] Correr §8. `ruff` + `pytest` verdes; `git status` sin cambios accidentales.

---

## 8. Comandos de validación (sin entrenamiento)

```powershell
# Defaults canónicos (fuente de verdad)
rg -n "gamma|net_arch|n_quantiles|exploration_fraction|REWARD_CONFIG" src/train_rl_defender.py

# Claims stale residuales en docs tracked (.md y .tex)
rg --no-ignore -n "gamma=0.99|net_arch=\[?512|exploration_fraction.*0.005" "docs/Personal Research"

# Claim falso fp=-1.5 (solo en graphify-out stale; debe desaparecer al regenerar)
rg --no-ignore -n "fp=-1.5|Current \(fp=-1.5\)"

# Rutas Phase 2 viejas sin _MAIN (deben quedar 0 tras R3)
rg -n "P2v2_pred_20260610_161231(?!_MAIN)" docs experiments README.md

# Referencias antes de archivar probes
rg -n "C01_|C02_|C03_|A01_|A02_|rl_defender_dqn|MAIN_.*fast_random" --glob '!graphify-out' --glob '!docs/audits'

# Enlace roto a fichero ignorado
git check-ignore "docs/Personal Research/deep-defense-research/README.md"

# Optuna
rg -n "optuna" pyproject.toml requirements.txt

# Estado / lint / tests
git status --short; uv run ruff check .; uv run pytest tests/
```

---

## 9. No tocar todavía
- **`datasets/`, `memoria/`, run+modelo MAIN (`models/MAIN_*full_*193655.zip`)** — evidencia oficial viva.
- **`src/` (lógica), `tests/`, CI** — sin código muerto; fuera del alcance de limpieza.
- **Bug fast-preset** (`max_rows` carga Lunes-benigno primero → learn-to-PERMIT; `load_cicids2017.py:535` `fast=100_000`): heredado del `tfg_cyber_ai_audit` (retirado). Corrección de **código**, no limpieza — track aparte.
- **`.tex/.pdf` compilados** — no recompilar hasta resolver §6.
- **Historia git** — solo `.gitignore` + `git rm --cached`; nada de reescritura.
- **Experimentos secundarios** — aún no existen; al crearlos, comparten test fijo y perfil `main-experiment`.

---

## 10. Reorg de `docs/` ya ejecutada (2026-06-25)
- Creado `docs/audits/` (con `README.md` índice). Esta guía vive ahí; los dos audits previos se consolidaron en §11 y se eliminaron.
- Creado `docs/archive/` con `informe.tex` / `informe.pdf` (borrador obsoleto de la memoria) (+ `README.md`).
- `docs/README.md` y `README.md` (raíz) actualizados: tesis canónica = `memoria/`; `report/` (EN) aparcada; índice de auditorías.
- **Personal Research (§6.2 resuelto):** se conserva solo `.md`; eliminados `.tex`+`.pdf` derivados; `.gitignore` ya no ignora los `.md`; enlace de `docs/README.md` apuntado al `.md`. Marcadas como investigación, no fuente de verdad.
- **Pendiente (requiere estar al teclado, ver Fases 3/5):** archivar probes de `runs/`/`models/`, destrackear binarios pesados de `runs/`.

---

## 11. Carry-forward de auditorías previas (consolidado 2026-06-25)

Los dos audits previos se revisaron contra el estado actual y se **retiraron** (historia en git). Resueltos descartados; ítems útiles volcados aquí.

**De `stale_claims_diagnosis_2026-06-15.md`:**
- Resueltos (descartados): hiperparámetros en Personal Research (reencuadrados como investigación, §6.2); warning de `--timesteps`; "C03 como artefacto actual" (DEFENSA ya dice "histórico"); `.md` ignorado (§6.2 / H1).
- Vivos: **3.1 → D6** (Phase 2 en DEFENSA), **5.1 → D4** (MAIN en results.md), **5.2 → R4** (z-score, menor).

**De `tfg_cyber_ai_audit.md`:**
- Resueltos/descartados: overclaiming de la introducción (no era defecto — decisión deliberada, §6.3).
- Vivos: **optuna → C1**, **dataset SHA/descarga → C2**, **fast-preset → §9** (código, aparte), **`.gitignore` global → H2 / Fase 5**, **terminología Check A/B/C → D7** (baja).

Resultado: esta guía es el **único audit vivo**; todo lo pendiente está en §3 + Fases.
