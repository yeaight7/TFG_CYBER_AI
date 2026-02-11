# AGENT_CONTEXT (TFG_CYBER_AI)

## Objetivo del TFG

Desarrollar un **agente defensor de ciberseguridad basado en Aprendizaje por Refuerzo (Reinforcement Learning)** capaz de clasificar tráfico de red como benigno o ataque, y decidir acciones óptimas (PERMIT/BLOCK) para maximizar la seguridad minimizando falsos positivos y negativos.

### Fases del Proyecto

**Fase 1 — Clasificación/Detección sobre Datasets**:
- Entrenar agente RL sobre datasets históricos y modernos
- Comparar con baselines supervisados (Random Forest)
- Definir esquema canónico de features
- Validar funcionamiento del framework RL

**Fase 2 — Entorno Simulado con Tráfico Generado**:
- Generar tráfico real (benigno + ataques) con herramientas (Kali, scripts)
- Capturar tráfico (PCAP)
- Extraer features de flujos (flow features) en tiempo real
- Agente RL decide PERMIT/BLOCK sobre tráfico capturado
- Almacenar dataset de simulación (features + acción + label ground-truth)

---

## Cambio de Trayectoria (Decisión Fundamental)

**IMPORTANTE**: El proyecto adoptó un cambio de trayectoria crítico para garantizar coherencia y reproducibilidad.

### Problema Original

Entrenar con múltiples datasets que tienen **columnas/features diferentes** crea varios problemas:
1. El modelo RL necesita un espacio de observación **fijo** (misma longitud, mismo significado por posición)
2. Si entrenamos con `[a,b,c,d,e]` y luego en simulación tenemos `[a,b,c,x,y]`, el modelo falla
3. Comparaciones entre datasets son injustas (bias según features disponibles)
4. En simulación, no sabríamos qué features extraer del tráfico real

### Solución: Esquema Canónico de Features (FEATURES_CANON)

Definir un **conjunto fijo de features canónicas** que:
1. Sirva de "lenguaje común" para TODOS los datasets
2. Sea extraíble de tráfico real/PCAP en simulación (flow-based features)
3. Permita entrenar el agente con múltiples datasets de forma justa
4. Garantice que el agente ve siempre el mismo espacio de observación

**Todos los datasets deben convertirse al esquema canónico mediante adapters.**

---

## Regla de "Justicia" (Fairness)

Para garantizar comparaciones válidas entre modelos y datasets:

- ✅ **Mismo conjunto de features canónicas** (mismo orden, mismo significado)
- ✅ **Mismo preprocesado** (manejo de missingness, scaling)
- ✅ **Mismas métricas y costes** (FP/FN, sistema de recompensas)
- ✅ **Misma semilla y estrategia de split** (reproducibilidad)

**Consecuencia**: No podemos simplemente concatenar datasets con columnas diferentes. Debemos mapear primero al esquema canónico.

---

## Datasets

El proyecto está diseñado para trabajar con **MÚLTIPLES datasets** (no solo 2). Cada dataset pasa por un **adapter** que lo convierte al esquema canónico.

### Datasets Actuales

#### 1. NSL-KDD (Benchmark Histórico)
- **Kaggle**: `hassan06/nslkdd`
- **Propósito**: Benchmark histórico para Fase 1, demostración del framework RL
- **Características**: 
  - 41 features originales (basadas en conexiones antiguas)
  - Features muy diferentes a datasets modernos flow-based
  - Dataset de 1999, tráfico simulado de laboratorio
- **Uso en el proyecto**: 
  - ✅ Fase 1: Warm-up, comparación con literatura, prueba del framework
  - ❌ **NO** forma parte del modelo final para simulación
  - ❌ **NO** se usa para definir el esquema canónico
- **Razón**: Sus features son incompatibles con flow extractors modernos y no se pueden extraer de PCAP en simulación

#### 2. CICIDS2017 (Dataset Principal Moderno)
- **Kaggle**: `chethuhn/network-intrusion-dataset`
- **Propósito**: Dataset principal que **define el esquema canónico**
- **Características**:
  - ~80 features flow-based (extraídas con CICFlowMeter)
  - Tráfico moderno (2017), incluye ataques web, botnets, DDoS, etc.
  - 2.8M+ muestras
- **Uso en el proyecto**:
  - ✅ Base para definir `FEATURES_CANON`
  - ✅ Entrenamiento de modelos para Fase 2
  - ✅ Referencia para transición a simulación

#### 3. Datasets Futuros (Extensibilidad)

El diseño soporta añadir **N datasets adicionales** mediante adapters:

- **Candidatos ideales**: Familia CIC (CICIDS2018, CSE-CIC-IDS2018), UNSW-NB15, otros con flow features
- **Requisito**: Que sus features sean compatibles o mapeables al esquema canónico
- **Proceso**: Crear `load_<nombre_dataset>.py` que mapee al esquema canónico + máscara de missingness

**Nota importante**: El valor del multi-dataset training es mayor cuando los datasets comparten el mismo **tipo de features** (flow-based). Mezclar datasets incompatibles (como NSL-KDD con CICIDS) requiere mapeos artificiales que pueden introducir ruido.

---

## Features Canónicas (FEATURES_CANON)

### Estado Actual: Pendiente de Definición Formal

Las features canónicas se están definiendo. Criterios de selección:

1. **Existen en CICIDS2017** (dataset principal moderno)
2. **Extraíbles de tráfico real/PCAP** en simulación mediante flow extractors (CICFlowMeter, Zeek, etc.)
3. **Sin data leakage**: NO incluir IPs, timestamps absolutos, Flow IDs, puertos específicos
4. **Preferiblemente numéricas**: Facilita el aprendizaje RL
5. **Estables y robustas**: No dependen de peculiaridades del dataset

### Categorías de Features Canónicas (Borrador)

Basadas en análisis de CICIDS2017:

- **Estadísticas de flujo**: duración, número de paquetes/bytes (fwd/bwd)
- **Tasas y velocidades**: pkt/s, bytes/s
- **Estadísticas de tamaños**: mean, std, min, max de tamaños de paquetes
- **Estadísticas de tiempos**: inter-arrival times (IAT), tiempos activos/idle
- **Flags TCP**: contadores de SYN, ACK, FIN, RST, etc.
- **Ventanas TCP**: inicial, mean, etc.
- **Estadísticas derivadas**: ratios, entropías, etc.

### Máscara de Missingness (Missingness Mask)

**CRÍTICO**: Cuando una feature canónica no existe en un dataset:

1. **NO** poner simplemente 0 (puede confundirse con un valor real de 0)
2. **Imputar** con valor razonable (0, media, mediana según contexto)
3. **Marcar** con máscara de missingness: `m_i = 0` (0 = ausente, 1 = presente)

**Vector de observación final**:
```
obs = [x_1, x_2, ..., x_d, m_1, m_2, ..., m_d]
```

Donde:
- `x_i` = valor de la feature i (imputado si no existe)
- `m_i` = 1 si la feature estaba presente, 0 si fue imputada

Esto permite al agente RL **aprender qué features son confiables** para cada tipo de dato.

---

## Pipeline de Simulación (Visión Fase 2)

```
┌─────────────────────────────────────────────────────────────┐
│ 1. GENERACIÓN DE TRÁFICO                                    │
│    - Kali Linux con herramientas de ataque                  │
│    - Scripts de tráfico benigno (navegación, descargas)     │
│    - Ataques: DDoS, port scans, exploits, etc.              │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. CAPTURA DE TRÁFICO (PCAP)                                │
│    - tcpdump / Wireshark / tshark                           │
│    - Captura de paquetes en tiempo real                     │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. EXTRACCIÓN DE FEATURES (Flow Features)                   │
│    - CICFlowMeter / Zeek / custom extractor                 │
│    - Produce vectores en formato FEATURES_CANON             │
│    - Output: [x_1, ..., x_d, m_1, ..., m_d]                │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. AGENTE RL DECIDE (PERMIT/BLOCK)                          │
│    - Lee vector de features del flujo                       │
│    - Acción 0 = PERMIT (permitir)                           │
│    - Acción 1 = BLOCK (bloquear)                            │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. ALMACENAR DATASET DE SIMULACIÓN                          │
│    - CSV/Parquet con: features + acción + label truth      │
│    - Label truth: conocemos qué tráfico generamos (0/1)     │
│    - Permite evaluar decisiones del agente                  │
└─────────────────────────────────────────────────────────────┘
```

### Pregunta Pendiente de Diseño

**¿Qué feature extractor usar en simulación?**
- Opción A: **CICFlowMeter-like** (mismo que CICIDS2017, compatibilidad directa)
- Opción B: **Zeek-like** (más flexible, ampliamente usado en industria)
- Opción C: **Custom extractor** (máximo control, más trabajo)

Decisión: **Pendiente de evaluación en Fase 2**. Probablemente CICFlowMeter para máxima compatibilidad con CICIDS2017.

---

## Estado Actual del Proyecto (Implementado)

### Archivos y Componentes Existentes

#### `src/load_nsl_kdd.py`
- ✅ Loader para NSL-KDD desde Kaggle (kagglehub)
- ✅ Preprocesamiento: one-hot encoding, etiquetas binarias
- ✅ Retorna: `(X_train, y_train, X_test, y_test)`
- ⚠️ **Limitación**: No está adaptado al esquema canónico (todavía)

#### `src/load_cicids2017.py`
- ✅ Loader para CICIDS2017 desde Kaggle
- ✅ Limpieza de datos: eliminación de IPs, timestamps, Flow IDs
- ✅ Manejo de NaNs e infinitos
- ✅ Split estratificado, scaling opcional
- ✅ Retorna: `(X_train, y_train, X_test, y_test, scaler, feature_names)`
- ⚠️ **Limitación**: No está adaptado al esquema canónico (todavía)

#### `src/rl_defender_env.py`
- ✅ Entorno RL custom (Gymnasium)
- ✅ Espacio de observación: vector de features (Box)
- ✅ Espacio de acciones: Discreto(2) — 0=PERMIT, 1=BLOCK
- ✅ Sistema de recompensas configurable (TP/FP/FN/TN)
- ✅ Soporte para shuffle y episodios limitados

#### `src/train_rl_defender.py`
- ✅ Script de entrenamiento DQN con Stable-Baselines3
- ✅ Generación automática de RUN_ID con timestamp
- ✅ Logging a TensorBoard (`runs/nslkdd/<RUN_ID>/`)
- ✅ Evaluación en test con confusion matrix y classification report
- ✅ Guardado de modelos en `models/<RUN_ID>.zip`

#### `src/baseline_random_forest.py`
- ✅ Baseline supervisado con Random Forest
- ✅ Evaluación comparable con métricas de clasificación
- ✅ Guardado de modelo en `models/rf_nslkdd.joblib`

### Directorios y Estructura

```
TFG_CYBER_AI/
├── datasets/          — NSL-KDD descargado automáticamente
├── docs/              — Documentación, contexto, decisiones
├── experiments/       — Tracking de experimentos (pendiente)
├── models/            — Modelos guardados (.zip, .joblib)
├── runs/              — Resultados con RUN_ID (TensorBoard logs)
└── src/               — Código fuente Python
```

---

## Próximos Pasos Inmediatos

### 1. Auditoría de Columnas CICIDS2017

- [ ] Ejecutar `load_cicids2017.py` y listar todas las columnas disponibles
- [ ] Clasificar columnas en: flow features, metadata, labels
- [ ] Identificar columnas de leakage (ya eliminadas, verificar)
- [ ] Documentar significado de cada feature

### 2. Definir FEATURES_CANON Formalmente

- [ ] Seleccionar ~30-80 features flow-based de CICIDS2017
- [ ] Documentar criterio de selección para cada feature
- [ ] Crear lista `FEATURES_CANON` en código (ej. `src/canonical_schema.py`)
- [ ] Verificar que todas son extraíbles de PCAP

### 3. Implementar Adapters con Esquema Canónico

- [ ] Modificar `load_cicids2017.py` para mapear a `FEATURES_CANON`
- [ ] Implementar máscara de missingness
- [ ] Modificar `load_nsl_kdd.py` para mapear a `FEATURES_CANON` (o marcarlo como legacy/benchmark)
- [ ] Vector final: `[x_1..x_d, m_1..m_d]`

### 4. Entrenar Modelos con Esquema Canónico

- [ ] Re-entrenar Random Forest sobre CICIDS2017 con esquema canónico
- [ ] Re-entrenar DQN sobre CICIDS2017 con esquema canónico
- [ ] Comparar métricas (accuracy, precision, recall, F1)
- [ ] Documentar resultados en `experiments/cicids2017_canonical.md`

### 5. Preparar para Multi-Dataset Training

- [ ] Validar que adapters producen el mismo formato de salida
- [ ] Implementar función de "merge" de datasets con esquema canónico
- [ ] Verificar que la máscara de missingness funciona correctamente
- [ ] Experimentar con entrenamiento combinado (si hay múltiples datasets disponibles)

---

## Decisiones Técnicas Pendientes

### 1. Feature Extractor para Simulación
- **Opciones**: CICFlowMeter, Zeek, custom
- **Decisión**: Pendiente de Fase 2
- **Recomendación actual**: CICFlowMeter (compatibilidad directa con CICIDS2017)

### 2. Estrategia de Imputación en Máscara de Missingness
- **Opciones**: 0, media, mediana, forward-fill, modelo de imputación
- **Decisión**: Pendiente de definir por feature
- **Recomendación actual**: 0 para contadores, media para estadísticas, según contexto

### 3. Número Final de Features Canónicas
- **Rango**: 30-80 features
- **Decisión**: Pendiente de auditoría de CICIDS2017
- **Trade-off**: Más features = más información, pero más complejidad y riesgo de overfitting

### 4. Inclusión de NSL-KDD en Modelo Final
- **Opciones**: Incluir con adapter, excluir del modelo final
- **Decisión**: **Excluir del modelo final para simulación**
- **Razón**: Features incompatibles con flow extractors modernos
- **Uso**: Solo como benchmark histórico en Fase 1

---

## Referencias y Recursos

- **Repositorio**: https://github.com/yeaight7/TFG_CYBER_AI
- **Datasets**:
  - NSL-KDD: `hassan06/nslkdd` (Kaggle)
  - CICIDS2017: `chethuhn/network-intrusion-dataset` (Kaggle)
- **Frameworks**:
  - RL: Stable-Baselines3 (DQN, PPO, A2C)
  - Entorno: Gymnasium
  - ML: scikit-learn
- **Documentación interna**:
  - `docs/discusion_con_llm.md`: Historial de decisiones (NO MODIFICAR)
  - `.github/copilot-instructions.md`: Convenciones de código
  - `AGENTS.md`: Checklist para coding agents

---

## Notas Finales

Este documento es la **fuente de verdad** del proyecto. Si hay contradicciones con otros documentos, este prevalece (excepto `discusion_con_llm.md` que es histórico).

Actualiza este documento cuando:
- Se defina formalmente `FEATURES_CANON`
- Se implementen adapters al esquema canónico
- Se completen pasos de "Próximos pasos inmediatos"
- Cambien decisiones de diseño fundamentales
- Se añadan nuevos datasets

**Última actualización**: 2026-02-10 (creación de documentación para GitHub Copilot)
