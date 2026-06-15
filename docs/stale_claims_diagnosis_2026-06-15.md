# Diagnostico de stale claims en documentacion TFG

Fecha: 2026-06-15

Alcance: auditoria read-only del estado actual del codigo, `docs/results.md`, artefactos JSON pequenos citados en `runs/`, y Markdown espanol orientado a tesis/defensa.

No se auditaron ni tocaron `.tex`, `.pdf`, datasets, PCAPs, entrenamientos pesados, `memoria/`, logs ni recorridos completos de `runs/`.

## Resumen ejecutivo

- La fuente tecnica actual esta alineada en rewards: `tp=1.5`, `fp=-2.0`, `fn=-5.0`, `omission=0.0`.
- La discrepancia mas importante esta en docs de investigacion/defensa que todavia describen hiperparametros antiguos o teoricos como si fueran el estado actual del repo.
- `C03` debe presentarse como resultado historico fuerte en random split, no como run principal actual. El run principal actual es `MAIN_qrdqn_cicids2017_canonical_full_random_20260609_193655`.
- `docs/results.md` coincide numericamente con los JSON citados, pero omite `MAIN` en la lista final de ubicaciones de training artifacts.
- `docs/Personal Research/**/*.md` esta ignorado por `.gitignore`; cualquier correccion ahi no sera versionada salvo cambio explicito de trackability.

## Findings priorizados

### 1. Claims objetivamente falsos sobre codigo actual

#### 1.1 Hiperparametros QRDQN actuales mal descritos

Archivos afectados:

- `docs/Personal Research/qrdqn-research-report.md`
- `docs/Personal Research/deep-defense-research/03-entorno-rl-algoritmo-y-entrenamiento.md`
- `docs/Personal Research/models-parameters-and-validation-thesis-defense-report.md`

Claim problematico:

- Se describe `gamma=0.99` como valor actual.
- Se presenta `net_arch=[512,256]` como arquitectura actual general.
- Se trata `n_quantiles=200` y la exploracion como defaults implicitos/heredados, no declarados.

Fuente de verdad:

- `src/train_rl_defender.py`
- `runs/cicids2017/MAIN_qrdqn_cicids2017_canonical_full_random_20260609_193655/config.json`

Verdad actual:

- Perfil `main-experiment`:
  - `net_arch=[1024,1024,512]`
  - `n_quantiles=200`
  - `learning_rate=5e-5`
  - `gamma=0.0`
  - `exploration_initial_eps=1.0`
  - `exploration_final_eps=0.02`
  - `exploration_fraction=0.10`
  - `total_timesteps=3_000_000`
- Perfil no-main:
  - `net_arch=[512,256]`
  - `n_quantiles=200`
  - `learning_rate=1e-4`
  - `gamma=0.0`
  - `exploration_final_eps=0.01`
  - `exploration_fraction=0.005`

Correccion propuesta:

> En el codigo actual, `gamma` esta fijado a `0.0`, no a `0.99`. El perfil `main-experiment` usado por el run MAIN declara explicitamente `net_arch=[1024,1024,512]`, `n_quantiles=200` y parametros de exploracion. La arquitectura `[512,256]` corresponde al perfil no-main y a runs historicos como C03, no al perfil principal actual.

#### 1.2 Warning obsoleto sobre help de `--timesteps`

Archivos afectados:

- `docs/Personal Research/data-structure-and-canonical-schema-research-report.md`
- `docs/Personal Research/qrdqn-research-report.md`

Claim problematico:

- Se afirma que el texto de ayuda CLI sobre timesteps sigue desalineado con el codigo.

Fuente de verdad:

- `src/train_rl_defender.py`

Verdad actual:

- El help actual ya describe `25k fast`, `100k full` y `3M main-experiment`.

Correccion propuesta:

> Eliminar la advertencia o marcarla como historica. El CLI actual ya refleja los defaults efectivos de timesteps.

### 2. Claims que mezclan resultados historicos y defaults actuales

#### 2.1 `C03` tratado como mejor artefacto actual

Archivos afectados:

- `docs/DEFENSA_TFG_SCRIPT.md`
- `docs/Personal Research/data-structure-and-canonical-schema-research-report.md`
- `docs/Personal Research/qrdqn-research-report.md`

Claim problematico:

- `C03_qrdqn_cicids2017_canonical_full_random_20260223_232439` aparece como mejor artefacto comprometido o referencia principal actual.

Fuente de verdad:

- `.github/AGENT_CONTEXT.md`
- `docs/results.md`
- `runs/cicids2017/MAIN_qrdqn_cicids2017_canonical_full_random_20260609_193655/config.json`
- `runs/cicids2017/C03_qrdqn_cicids2017_canonical_full_random_20260223_232439/config.json`

Verdad actual:

- `C03` es historico y obtuvo metricas mejores en un random split con `max_rows=500000` y test de 100,000 filas con mezcla distorsionada.
- `MAIN` es el run principal actual, entrenado con preset `full`, `3,000,000` timesteps y test partition completa de 566,149 filas.
- C03 y MAIN no son directamente comparables.

Correccion propuesta:

> C03 sigue siendo el mejor resultado historico en random split por metrica puntual, pero MAIN es el run principal actual y la referencia para el estado mantenido del proyecto. C03 uso `max_rows=500000` y un test de 100,000 filas; MAIN usa el conjunto completo y una particion de test de 566,149 filas. No deben compararse como si midieran el mismo protocolo.

### 3. Claims que mezclan benchmark CICIDS2017 con Phase 2

#### 3.1 Fase 2 descrita solo con runs benign-only antiguos

Archivos afectados:

- `docs/DEFENSA_TFG_SCRIPT.md`
- `docs/DEFENSA_TFG_PROGRESO.md`
- `docs/Personal Research/data-structure-and-canonical-schema-research-report.md`

Claim problematico:

- La narrativa de Fase 2 se apoya casi solo en:
  - `P2v2_pred_20260224_004121`: `block_rate=1.0`
  - `P2v2_pred_20260408_230318`: `block_rate=0.0`

Fuente de verdad:

- `docs/results.md`
- `runs/phase2/P2v2_pred_20260610_161231/config.json`
- `runs/phase2/P2v2_pred_20260610_161231/metrics.json`

Verdad actual:

- Existe un artefacto posterior etiquetado sobre `pcaps/synthetic_real_traffic.csv` con modelo MAIN:
  - `block_rate=0.252364`
  - `accuracy=0.991862`
  - `precision_attack=0.9791927533`
  - `recall_attack=0.988452`
  - `f1_attack=0.9838005908`

Correccion propuesta:

> Mantener los dos runs benign-only como evidencia de sensibilidad y domain shift, pero anadir el run `P2v2_pred_20260610_161231` como benchmark sintetico etiquetado de Phase 2. Dejar explicito que no debe mezclarse con el benchmark interno CICIDS2017.

### 4. Problemas de trackability por `.gitignore`

#### 4.1 Markdown de `docs/Personal Research` ignorado

Archivos afectados:

- `docs/Personal Research/*.md`
- `docs/Personal Research/deep-defense-research/*.md`
- `.gitignore`

Evidencia:

- `.gitignore` ignora explicitamente:
  - `docs/Personal Research/*.md`
  - `docs/Personal Research/deep-defense-research/*.md`

Riesgo:

- Correcciones en esos documentos no apareceran en `git status`.
- No entraran en commits.
- Pueden usarse como fuente de defensa sin trazabilidad real.

Correccion propuesta:

- Si son fuente de verdad para tesis/defensa: mover contenido final a una ruta trackeada o anadir excepciones en `.gitignore`.
- Si son borradores locales: mantenerlos ignorados, pero no basar claims versionados en ellos sin copiar la version final a una ruta trackeada.

### 5. Mejoras de claridad/estructura

#### 5.1 `docs/results.md` omite MAIN en artifact locations

Archivo afectado:

- `docs/results.md`

Problema:

- La seccion `Artifact Locations > Training` lista C01/C02/C03, pero omite `MAIN_qrdqn_cicids2017_canonical_full_random_20260609_193655`, aunque el documento lo usa arriba como run principal.

Correccion propuesta:

> Anadir `runs/cicids2017/MAIN_qrdqn_cicids2017_canonical_full_random_20260609_193655/` en la lista de training artifacts.

#### 5.2 Matiz de diagnosticos z-score en Phase 2

Archivos potenciales:

- `docs/DEFENSA_TFG_PROGRESO.md`
- `docs/DEFENSA_TFG_SCRIPT.md`
- `docs/Personal Research/deep-defense-research/05-phase2-laboratorio-inferencia-y-riesgos.md`
- `docs/Personal Research/deep-defense-research/07-artefactos-scripts-tests-y-validacion.md`

Matiz tecnico:

- En `scripts/predict_real_traffic_v2.py`, `compute_diagnostics()` se ejecuta despues de `apply_z_clipping()` cuando `--clip-z` esta activo.
- Por tanto, `z_abs_max` y `z_gt10_*` describen el estado post-clipping en esos runs, no necesariamente el drift bruto antes del clipping.

Correccion propuesta:

> Cuando `--clip-z` esta activo, los diagnosticos z-score exportados describen la distribucion tras el clipping. Para analizar drift bruto pre-clipping haria falta capturarlo explicitamente antes de aplicar `apply_z_clipping`.

## Documentacion correcta que conviene conservar

- Reward actual bien descrito: `tp=1.5`, `fp=-2.0`, `fn=-5.0`, `omission=0.0`.
- Distincion correcta entre rewards actuales y runs historicos: C03 coincide con el reward actual; C01/C02 usaron `fp=-1.0`.
- `leave-one-exact-CSV-out` esta correctamente descrito como implementado en codigo pero sin artefacto completo comprometido.
- Phase 2 esta correctamente planteada como inferencia offline, no bloqueo activo ni despliegue productivo.
- `docs/results.md` coincide numericamente con los JSON citados para MAIN, C03, Checks A/B/C y los runs Phase 2 auditados.

## Archivos a tocar en una fase de correccion

Trackeados:

- `docs/DEFENSA_TFG_SCRIPT.md`
- `docs/DEFENSA_TFG_PROGRESO.md`
- `docs/results.md`

Ignorados, tocar solo si se decide resolver trackability:

- `docs/Personal Research/qrdqn-research-report.md`
- `docs/Personal Research/data-structure-and-canonical-schema-research-report.md`
- `docs/Personal Research/models-parameters-and-validation-thesis-defense-report.md`
- `docs/Personal Research/deep-defense-research/03-entorno-rl-algoritmo-y-entrenamiento.md`
- `docs/Personal Research/deep-defense-research/05-phase2-laboratorio-inferencia-y-riesgos.md`
- `docs/Personal Research/deep-defense-research/07-artefactos-scripts-tests-y-validacion.md`

No tocar:

- `.pdf`
- `.tex`
- datasets
- PCAPs
- entrenamiento pesado
- `memoria/`
- logs
- recorridos completos de `runs/`

## Validacion recomendada despues de corregir

```powershell
git diff -- docs/DEFENSA_TFG_PROGRESO.md docs/DEFENSA_TFG_SCRIPT.md docs/results.md
rg --no-ignore --glob "*.md" -n "gamma=0.99|gamma = 0.99|n_quantiles no se fija|exploracion no se fija|exploration_fraction.*default|mejor artefacto.*C03|mejor run.*C03" docs
rg --no-ignore --glob "*.md" -n "MAIN_qrdqn_cicids2017_canonical_full_random_20260609_193655|P2v2_pred_20260610_161231|synthetic_real_traffic|leave-one-exact-CSV-out" docs
git status --short
```

No ejecutar entrenamiento pesado para esta correccion documental.
