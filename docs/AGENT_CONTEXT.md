# AGENT_CONTEXT (TFG_CYBER_AI)

## Objetivo del TFG
Agente defensor con RL para ciberseguridad.
Fase 1: clasificación/detección sobre datasets.
Fase 2: entorno simulado (tráfico generado) + extracción de features + decisiones del agente.

## Cambio de trayectoria (importante)
Objetivo: definir un conjunto de FEATURES “canónico” y usarlo siempre:
- Para comparar datasets de forma justa (misma representación/variables).
- Para poder generar tráfico en simulación y extraer EXACTAMENTE esas features.

## Regla de “justicia” (fairness)
Las comparaciones entre modelos/datasets se harán:
- con el mismo conjunto de features canónicas (mismo orden y definición),
- mismo preprocesado (missing, scaling),
- mismas métricas y coste (FP/FN),
- misma semilla y estrategia de split documentada.

## Datasets
- NSL-KDD: se usa como benchmark inicial (features antiguas/propias).
- CICIDS2017 (Kaggle: chethuhn/network-intrusion-dataset): dataset principal moderno para fijar features canónicas y transición a simulación.

## Features canónicas (pendiente de definir)
Elegidas para:
1) existir en CICIDS2017,
2) poder extraerse de tráfico real/pcap en simulación (flow features),
3) ser estables y no incluir identificadores/leakage (IPs, timestamps, flow_id).

## Pipeline de simulación (visión)
Generator (Kali/attacks + benign) ->
Capture (pcap) ->
Feature extraction (flow features) ->
RL agent decide PERMIT/BLOCK ->
Store dataset de simulación (features + acción + label ground-truth)

## Estado actual (qué hay implementado)
- Loader NSL-KDD binario
- Entorno RL dataset-as-env (PERMIT/BLOCK + reward config)
- Baseline RF y DQN con logging y RUN_ID
- Experiments tracking

## Próximos pasos inmediatos
1) Auditoría de columnas/features CICIDS2017 y selección de features canónicas
2) Implementar loader CICIDS2017 + limpieza
3) Implementar “feature mapping” a esquema canónico
4) Entrenar RF + DQN/QRDQN sobre CICIDS2017 con esquema canónico
