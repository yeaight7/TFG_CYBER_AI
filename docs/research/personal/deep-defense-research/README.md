# Investigación profunda para defensa (paquete multiarchivo)

Este paquete está pensado para preparar una defensa completa del proyecto **TFG_CYBER_AI** ante un tribunal y también para explicarlo a una persona sin base previa.

## Objetivo de este paquete

- cubrir de forma rigurosa los temas clave técnicos y metodológicos
- explicar cada parte con lenguaje accesible
- separar claramente lo **implementado**, lo **histórico** y lo **pendiente**
- evitar afirmaciones sin respaldo en código o artefactos de `runs/`

## Relación con tus investigaciones previas

Ya existen dos investigaciones profundas:

1. `docs/research/personal/data-structure-and-canonical-schema-research-report.md`
2. `docs/research/personal/qrdqn-research-report.md`

Este nuevo paquete las **complementa** y las integra en una narrativa de defensa completa.

## Orden recomendado de lectura

1. `01-fundamentos-y-objetivo.md`
2. `02-datos-esquema-canonico-y-preprocesado.md`
3. `03-entorno-rl-algoritmo-y-entrenamiento.md`
4. `04-validacion-y-lectura-de-resultados.md`
5. `05-phase2-laboratorio-inferencia-y-riesgos.md`
6. `06-glosario-y-preguntas-tribunal.md`
7. `07-artefactos-scripts-tests-y-validacion.md`

## Mapa rápido de entradas técnicas del repositorio

- Esquema canónico: `src/canonical_schema.py`
- Carga CICIDS2017: `src/load_cicids2017.py`
- Entorno RL: `src/rl_defender_env.py`
- Entrenamiento QRDQN: `src/train_rl_defender.py`
- Validación A/B/C: `src/validate_checks.py`
- leave-one-CSV-out: `src/validate_leave_one_csv_out.py`
- Inferencia Phase 2 (mantenida): `scripts/predict_real_traffic_v2.py`
- Utilidades de clipping: `src/scaling_utils.py`
- Export TensorBoard: `scripts/export_tensorboard_scalars.py`

