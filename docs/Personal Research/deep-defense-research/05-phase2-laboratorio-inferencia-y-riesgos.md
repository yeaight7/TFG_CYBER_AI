# 05 — Phase 2: laboratorio, inferencia y riesgos abiertos

## 1) Qué es exactamente Phase 2 hoy

Phase 2 es **inferencia offline** sobre flujos extraídos de tráfico de laboratorio.

Entrada mantenida:

- `scripts/predict_real_traffic_v2.py`

No incluye todavía bloqueo activo inline.

## 2) Flujo operativo resumido

1. generar/capturar tráfico en lab privado
2. extraer flujos
3. mapear al esquema canónico
4. aplicar pipeline robusto
5. predecir PERMIT/BLOCK
6. guardar artefactos de run

## 3) Endurecimiento técnico de la v2

## Clipping por percentiles (pre-scaling)

- función: `apply_percentile_clipping` (`src/scaling_utils.py`)
- recorta features crudas a `[p_low, p_high]` aprendidos en train

## Clipping por z-score (post-scaling)

- función: `apply_z_clipping`
- limita valores escalados a `[-max_z, +max_z]`

## Diagnósticos

`predict_real_traffic_v2.py` calcula y exporta estadísticos de z-score para vigilar domain shift.

## 4) Artefactos esperados de cada run Phase 2

En `runs/phase2/<RUN_ID>/`:

- `config.json`
- `metrics.json`
- `predictions.csv`
- `diagnostics.json` (si se exporta)

## 5) Hallazgo central de Phase 2 (honesto)

Hay runs comprometidos con comportamiento muy distinto en benign-only. Esto obliga a evitar sobrepromesas y a citar siempre el `RUN_ID` al hacer afirmaciones.

Mensaje sólido:

- la pipeline está técnicamente lista
- la robustez en tráfico real sigue siendo el principal problema científico/ingenieril abierto

## 6) Laboratorio privado y seguridad operacional

Referencias: `docs/gcp_lab.md`, `docs/phase2_plan.md`, `lab/docker/*`

Principios:

- red aislada
- no escaneo fuera de lab
- control estricto de acceso SSH
- no exponer secretos ni datasets sensibles en repo

## 7) Qué faltaría para pasar a bloqueo activo

1. estabilidad consistente en benign, ataque y mixto
2. umbrales operativos de FP/FN aceptables por caso de uso
3. procedimiento de rollback y supervisión humana
4. pruebas de seguridad operacional antes de automatizar bloqueo

