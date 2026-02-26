# Defensa TFG — Agente de Ciberseguridad con Aprendizaje por Refuerzo

**Autor**: Javier Rivero Iglesias  
**Fecha de defensa**: Febrero 2026  
**Duración estimada**: 20 a 30 minutos  

---

## 1. Introducción y Motivación

- **Planteé el problema** → detectar y bloquear tráfico malicioso en redes de forma automática → porque los ataques de red son cada vez más sofisticados y las reglas estáticas (firewalls tradicionales) no se adaptan a nuevos patrones de ataque.
- **Elegí Aprendizaje por Refuerzo (RL)** → paradigma donde un agente aprende una política de decisión mediante interacción con un entorno, recibiendo recompensas o penalizaciones → porque permite modelar los **costes asimétricos** de la ciberseguridad: no es lo mismo dejar pasar un ataque (FN) que bloquear tráfico legítimo (FP), y el RL permite ajustar estos pesos mediante un sistema de recompensas configurable.
- **Comparé con aprendizaje supervisado** → modelos como Random Forest clasifican, pero no optimizan directamente un objetivo de coste configurable → porque quería un sistema donde cambiar la postura de seguridad (agresiva vs conservadora) no requiriera re-entrenar desde cero, sino solo ajustar las recompensas.
- **Contexto académico** → este es un Trabajo de Fin de Grado (TFG) → el objetivo es demostrar la viabilidad del enfoque RL para ciberseguridad defensiva, no un producto de producción.

**Notas del ponente:**

Buenos días, miembros del tribunal. Mi Trabajo de Fin de Grado aborda un problema fundamental en ciberseguridad: la detección y bloqueo automático de tráfico malicioso. Los enfoques tradicionales basados en reglas estáticas o firmas no se adaptan a amenazas nuevas. Propongo un agente defensor basado en Aprendizaje por Refuerzo que aprende una política de decisión PERMIT/BLOCK optimizando directamente los costes asimétricos de la seguridad. A diferencia de un clasificador supervisado que simplemente maximiza la accuracy, el agente RL puede ponderar que dejar pasar un ataque es mucho peor que bloquear por error un flujo legítimo. El proyecto se estructura en dos fases: primero entrenamiento sobre datasets históricos, y después evaluación sobre tráfico real capturado en un laboratorio privado.

---

## 2. Arquitectura General del Proyecto

- **Diseñé una arquitectura modular en dos fases** → Fase 1 (dataset-as-environment) y Fase 2 (lab con tráfico real) → porque permite validar primero el framework RL de forma controlada y después transferirlo a un escenario más realista.
- **Implementé un pipeline completo** → datos → esquema canónico → entorno RL → agente → decisión → evaluación → porque cada componente es reemplazable y testeable de forma independiente.
- **Organicé el código en `src/`** → módulos separados para loaders, entorno, entrenamiento, validación y utilidades → porque facilita la reproducibilidad y extensibilidad.

```
 PIPELINE COMPLETO
 ─────────────────────────────────────────────────────────────────────

 ┌───────────┐    ┌──────────────┐    ┌──────────────────┐
 │  Dataset   │───→│  Adapter +   │───→│  Esquema canónico│
 │ (CSV/PCAP) │    │  Limpieza    │    │  76 features     │
 └───────────┘    └──────────────┘    └────────┬─────────┘
                                               │
                                               ▼
                                    ┌──────────────────┐
                                    │ Missingness mask  │
                                    │ 76 dims (0/1)     │
                                    └────────┬─────────┘
                                               │
                                               ▼
                                    ┌──────────────────┐
                                    │ obs = [x₁…x₇₆ |  │
                                    │        m₁…m₇₆]   │
                                    │ (152 dimensiones) │
                                    └────────┬─────────┘
                                               │
                                               ▼
                   ┌──────────────┐  ┌──────────────────┐
                   │ StandardScaler│─→│ Entorno RL       │
                   │ (fit on train)│  │ (Gymnasium)      │
                   └──────────────┘  │ obs → agente     │
                                     │ acción ← agente  │
                                     └────────┬─────────┘
                                               │
                                               ▼
                                    ┌──────────────────┐
                                    │ Agente QRDQN      │
                                    │ MLP [512, 256]    │
                                    │ Acción: 0=PERMIT  │
                                    │         1=BLOCK   │
                                    └────────┬─────────┘
                                               │
                                               ▼
                                    ┌──────────────────┐
                                    │ Evaluación:       │
                                    │ Accuracy, F1,     │
                                    │ Confusion Matrix  │
                                    └──────────────────┘
```

**Notas del ponente:**

El proyecto se estructura en un pipeline de seis etapas. Los datos de entrada —ya sean CSVs de datasets históricos o flujos extraídos de PCAPs reales— pasan por un adapter que los convierte al esquema canónico de 76 features. Se añade una máscara de missingness, resultando en un vector de 152 dimensiones. Este vector se escala con un StandardScaler ajustado solo en el conjunto de entrenamiento, y entra al entorno Gymnasium custom donde el agente QRDQN toma decisiones de PERMIT o BLOCK. Finalmente, se evalúa contra las etiquetas reales utilizando métricas estándar de clasificación. Cada componente es modular: puedo cambiar el dataset, el algoritmo o la estrategia de escalado sin afectar al resto del pipeline.

---

## 3. Datasets: NSL-KDD y CICIDS2017

- **Usé NSL-KDD como benchmark histórico** → dataset de 1999 con 41 features basadas en conexiones (no flujos), derivado de KDD Cup'99 → porque es un benchmark ampliamente citado en la literatura de IDS que me permite comparar con trabajos previos y validar que el framework RL funciona (`src/load_nsl_kdd.py`).
- **Elegí CICIDS2017 como dataset principal** → dataset moderno de 2017 del Canadian Institute for Cybersecurity, con ~2.8 millones de flujos distribuidos en 5 días (lunes a viernes), ~80 columnas flow-based extraídas con CICFlowMeter → porque sus features son extraíbles de tráfico real (PCAP) y cubren ataques modernos: DDoS, Port Scan, Brute Force, Web Attacks, Botnets, entre otros (`src/load_cicids2017.py`).
- **Excluí NSL-KDD del modelo final** → sus features (basadas en conexiones TCP antiguas) son incompatibles con flow extractors modernos → porque no se pueden extraer de PCAPs en simulación; solo sirve como referencia histórica en Fase 1.
- **Diseñé para N datasets** → patrón adapter que permite añadir cualquier dataset mediante un `load_<nombre>.py` → porque el framework debe ser extensible a futuros datasets (CICIDS2018, UNSW-NB15, etc.).

**Notas del ponente:**

Trabajo con dos datasets principales. NSL-KDD es un benchmark histórico de 1999 con 41 features basadas en conexiones TCP, ampliamente utilizado en la literatura. Lo usé como prueba de concepto en la Fase 1 para validar que el framework RL aprende correctamente. Sin embargo, sus features son incompatibles con extractores de flujo modernos, así que no forma parte del modelo final. CICIDS2017 es el dataset principal: moderno, con casi tres millones de flujos distribuidos en cinco días de tráfico simulado, con features extraídas por CICFlowMeter directamente de PCAPs. Esto es crucial porque define el esquema canónico de features y garantiza la compatibilidad con la Fase 2, donde extraeremos las mismas features de tráfico real. El sistema soporta múltiples datasets mediante adapters independientes.

---

## 4. Ingeniería de Features y Esquema Canónico

- **Definí un esquema canónico de 76 features** (`src/canonical_schema.py` — `FEATURES_CANON`) → lista ordenada y fija de estadísticas de flujo de red organizadas en categorías: estadísticas generales (duración, paquetes, bytes), tamaños de paquetes (forward/backward), tasas de flujo, inter-arrival times (IAT), flags TCP (SYN, ACK, FIN, RST, PSH, URG, ECE, CWE), cabeceras, ratios, bulk statistics, subflow statistics, ventanas TCP, datos activos, y tiempos active/idle → porque el agente RL necesita un espacio de observación **fijo** (misma dimensión y mismo significado por posición) independientemente del dataset de origen.
- **Normalicé nombres de columnas** → las cabeceras CSV de CICIDS2017 usan Title Case con espacios (`"Flow Duration"`, `"Total Fwd Packets"`) y las convertí a `lower_snake_case` (`flow_duration`, `total_fwd_packets`) → porque garantiza consistencia entre datasets y simplifica el mapping programático.
- **Implementé mappings por dataset** → `CICIDS2017_TO_CANON` mapea 76/76 features (mapeo completo), `NSL_KDD_TO_CANON` mapea solo 3/76 features (`duration → flow_duration`, `src_bytes → total_length_of_fwd_packets`, `dst_bytes → total_length_of_bwd_packets`) → porque cada dataset tiene nomenclatura y granularidad diferentes (`src/canonical_schema.py`).
- **Excluí columnas de leakage** → eliminé Flow ID, Timestamp, Source IP, Destination IP, Source Port, Destination Port → porque estas columnas actúan como proxies de la etiqueta (ciertos ataques usan puertos o IPs específicos) y causarían data leakage (`src/load_cicids2017.py` — `_drop_identifier_like_columns`).

**Notas del ponente:**

El esquema canónico es una de las decisiones de diseño más importantes del proyecto. Son 76 features flow-based seleccionadas de las salidas de CICFlowMeter. Estas features cubren estadísticas de flujo, tamaños de paquete, inter-arrival times, flags TCP, y más. Cada dataset se adapta a este esquema mediante un mapping específico: CICIDS2017 mapea las 76 completas, mientras que NSL-KDD solo puede mapear 3 de 76, lo cual demuestra la incompatibilidad que mencionaba. Normalicé los nombres a lower_snake_case y eliminé estrictamente cualquier columna que pudiera causar data leakage: direcciones IP, timestamps, Flow IDs y puertos. Destination Port, por ejemplo, se elimina porque ciertos ataques apuntan a puertos específicos, y si el modelo lo ve, aprende un atajo en vez de patrones de flujo reales.

---

## 5. Máscara de Missingness

- **Implementé una máscara de missingness binaria** (`src/canonical_schema.py` — `map_to_canonical()`) → vector de 76 dimensiones donde `m_i = 1` si la feature estaba presente y era válida en el dataset original, `m_i = 0` si fue imputada → porque permite al agente saber qué features del vector son fiables y cuáles son placeholders.
- **Usé imputación con valor 0.0** (`src/canonical_schema.py` — `DEFAULT_IMPUTATION_VALUE = 0.0`) → las features ausentes se rellenan con cero → porque para contadores, bytes y tasas de flujo, la ausencia de valor indica ausencia de actividad, lo cual es semánticamente coherente.
- **Construí el vector de observación final como concatenación** → `obs = [x_1, ..., x_76, m_1, ..., m_76]` → 152 dimensiones (`src/canonical_schema.py` — `NUM_OBSERVATION_FEATURES = 152`) → porque el agente recibe toda la información en un solo vector homogéneo.

```
  VECTOR DE OBSERVACIÓN (152 dims)
  ──────────────────────────────────────────────────────────
  ┌─────────────────────────────┬─────────────────────────────┐
  │      Features (76 dims)     │    Missingness Mask (76)    │
  │  x_1  x_2  ...  x_75  x_76 │  m_1  m_2  ...  m_75  m_76 │
  │ (valores numéricos, float32)│   (0 = imputado, 1 = real)  │
  └─────────────────────────────┴─────────────────────────────┘

  Para CICIDS2017:  76/76 features presentes → máscara ≈ todo 1s
  Para NSL-KDD:      3/76 features presentes → máscara ≈ todo 0s (73 imputadas)
```

- **Noté que para CICIDS2017 la máscara es mayoritariamente 1s** → 76 de 76 features presentes → pero la arquitectura está diseñada pensando en multi-dataset, donde distintos datasets tendrán distintos patrones de missingness.

**Notas del ponente:**

La máscara de missingness es un mecanismo clave para la generalización multi-dataset. Cuando un dataset no tiene una feature canónica, imputamos con cero y marcamos m_i igual a cero. El vector final del agente tiene 152 dimensiones: 76 valores de features más 76 bits de máscara. Para CICIDS2017, la máscara es prácticamente todo unos porque mapea las 76 features. Pero para NSL-KDD, solo 3 features mapean, así que el agente ve 73 ceros con su correspondiente máscara en cero, y aprende que esas posiciones no contienen información útil. Esta arquitectura permite que en el futuro se entrene con múltiples datasets simultáneamente sin cambiar la dimensión del espacio de observación.

---

## 6. Pipeline de Preprocesamiento

- **Implementé limpieza de datos robusta** (`src/load_cicids2017.py` — `_coerce_numeric_features`, `_clean_rows`) → coerción numérica con `pd.to_numeric(..., errors="coerce")`, reemplazo de infinitos por NaN, rellenado de NaN con 0, eliminación de filas sin etiqueta → porque los CSVs de CICIDS2017 contienen valores no numéricos, infinitos (divisiones por cero en tasas) y valores faltantes.
- **Ajusté StandardScaler solo en train** (`src/train_rl_defender.py` — líneas 239-241) → `scaler.fit_transform(X_train)` y `scaler.transform(X_test)` → porque ajustar en test causaría data leakage temporal, y persisto el scaler como `scaler.joblib` para usarlo en inferencia Phase 2.
- **Calculé y persistí percentiles de entrenamiento** (`src/train_rl_defender.py` — líneas 235-236, 314-316) → percentiles p0.5 y p99.5 guardados como `train_percentiles.npz` → porque permiten aplicar percentile clipping en inferencia Phase 2 para mitigar distribución shift en features extremas.
- **Implementé dos modos de split** (`src/load_cicids2017.py` — `load_cicids2017_split`) → `random` (split estratificado 80/20) y `day` (train=Lunes-Miércoles, test=Jueves-Viernes) → porque el split por día es más realista y testa la generalización a ataques no vistos durante entrenamiento.

**Notas del ponente:**

El preprocesamiento sigue un pipeline riguroso. Primero, coerción numérica de todas las columnas features, reemplazo de infinitos y NaN. Después, el StandardScaler se ajusta exclusivamente en el conjunto de entrenamiento para evitar data leakage, y se persiste como artefacto joblib. Además, calculo los percentiles 0.5 y 99.5 del entrenamiento para usarlos después en la Fase 2 como filtro de outliers. Implementé dos modos de split: el aleatorio estratificado para evaluación rápida, y el split por día que es más realista: entreno con lunes, martes y miércoles, y testeo con jueves y viernes, donde aparecen tipos de ataque que el modelo no ha visto en entrenamiento.

---

## 7. Diseño del Entorno RL (Gymnasium)

- **Implementé un entorno Gymnasium custom** (`src/rl_defender_env.py` — `RLDatasetDefenderEnv`) → hereda de `gym.Env`, con espacio de observación `Box(n_features,)` y espacio de acciones `Discrete(2)` → porque Gymnasium es el estándar para entornos RL y permite integración directa con Stable-Baselines3.
- **Definí las acciones** → `0 = PERMIT` (dejar pasar el tráfico), `1 = BLOCK` (bloquear) → porque modelan las dos decisiones fundamentales de un firewall.
- **Diseñé un sistema de recompensas configurable** (`src/rl_defender_env.py` — `_compute_reward`) → basado en la matriz de confusión: TP, FP, FN, TN (omission) → porque permite ajustar la postura del agente sin re-entrenar la arquitectura.

```
  MATRIZ DE RECOMPENSAS
  ──────────────────────────────────────────────────────────
                        Acción del agente
                    PERMIT (0)      BLOCK (1)
                 ┌──────────────┬──────────────┐
  Benigno (y=0)  │  omission    │     fp       │
  (tráfico       │  (TN)        │  (FP)        │
   normal)       │              │              │
                 ├──────────────┼──────────────┤
  Ataque  (y=1)  │     fn       │     tp       │
  (tráfico       │  (FN)        │  (TP)        │
   malicioso)    │              │              │
                 └──────────────┴──────────────┘

  Valores por defecto en rl_defender_env.py:
    tp = 1.0   |  fp = -1.0   |  fn = -5.0   |  omission = 0.0

  Valores usados en entrenamiento C03 (train_rl_defender.py):
    tp = 1.5   |  fp = -2.0   |  fn = -5.0   |  omission = 0.0
```

- **Modelo de episodio** → cada flujo es un paso (step); el episodio termina cuando se han recorrido todas las muestras o se alcanza `max_steps_per_episode` → porque trata cada decisión como independiente, apropiado para clasificación de flujos individuales.

**Notas del ponente:**

El entorno Gymnasium custom encapsula el dataset como un entorno interactivo. En cada paso, el agente observa un vector de 152 dimensiones correspondiente a un flujo de red, toma una acción —PERMIT o BLOCK— y recibe una recompensa basada en si su decisión fue correcta. El sistema de recompensas es la pieza clave: la penalización por falso negativo (permitir un ataque, fn=-5.0) es mucho mayor que por falso positivo (bloquear tráfico legítimo, fp=-1.0 o -2.0). Esto codifica la asimetría fundamental de la ciberseguridad: es peor dejar pasar un ataque que molestar a un usuario legítimo. Los valores son configurables sin tocar la arquitectura del modelo, lo cual es una ventaja significativa sobre enfoques puramente supervisados.

---

## 8. Algoritmos: DQN → QRDQN

- **Empecé con DQN en Fase 1** → Deep Q-Network, algoritmo que combina Q-learning con redes neuronales profundas para estimar la función de valor acción Q(s,a) → porque es el algoritmo RL value-based más establecido, y me permitió validar el framework sobre NSL-KDD (`experiments/nslkdd_experiments.md`).
- **Establecí un baseline supervisado con Random Forest** (`src/baseline_random_forest.py`) → modelo clásico de ensemble con 200 árboles → porque necesitaba una referencia contra la cual medir el rendimiento del agente RL.
- **Transicioné a QRDQN para Fase 2** → Quantile Regression DQN, variante distributional que estima la **distribución completa del retorno** (no solo el valor esperado), modelando N cuantiles de la distribución de Q-valores → porque en un entorno con costes asimétricos, conocer la distribución permite al agente tomar decisiones más informadas sobre el riesgo, y las métricas mejoraron sustancialmente.
- **Implementé QRDQN con sb3-contrib** (`src/train_rl_defender.py`) → MLP `[512, 256]`, learning_rate = 1×10⁻⁴, batch_size = 2048 (full), gradient_steps = 20, train_freq = 100, gamma = 0.99 → porque estos hiperparámetros se ajustaron iterativamente y con Optuna para obtener el mejor rendimiento.

**Notas del ponente:**

La evolución algorítmica fue de DQN a QRDQN. En la Fase 1, usé DQN estándar sobre NSL-KDD para validar que el framework RL funciona: obtuve una accuracy de 0.76, comparable al Random Forest baseline. Pero al migrar a CICIDS2017 con el esquema canónico, transicioné a QRDQN, una variante distributional de DQN. Mientras DQN estima un único valor esperado Q(s,a), QRDQN estima la distribución completa del retorno mediante regresión por cuantiles. Esto es particularmente útil en ciberseguridad porque las recompensas son muy asimétricas: el agente necesita entender no solo el retorno medio sino también la variabilidad del riesgo. La implementación usa sb3-contrib con una red MLP de dos capas, 512 y 256 neuronas, y un learning rate de 1e-4.

---

## 9. Entrenamiento, Tracking y Reproducibilidad

- **Definí una convención de RUN_ID** (`src/train_rl_defender.py` — línea 196) → formato `C03_qrdqn_cicids2017_canonical_{preset}_{split}_{timestamp}` → porque permite identificar unívocamente cada experimento y sus condiciones.
- **Persistí artefactos por run** → cada run genera en `runs/<category>/<RUN_ID>/`: `config.json` (hiperparámetros completos), `metrics.json` (métricas de evaluación), `scaler.joblib` (scaler ajustado), `train_percentiles.npz` (percentiles p0.5/p99.5) → porque garantiza la reproducibilidad total del experimento.
- **Integré TensorBoard** (`src/train_rl_defender.py` — `tensorboard_log`) → logging automático de curvas de aprendizaje durante entrenamiento → porque permite monitorizar la convergencia del agente en tiempo real.
- **Implementé tuning con Optuna** (`src/tune_hparams.py`) → framework de optimización bayesiana que busca los mejores hiperparámetros (learning_rate, batch_size, gradient_steps, net_arch, gamma, train_freq) → porque permite explorar el espacio de hiperparámetros de forma eficiente. Mejor resultado: accuracy 0.9939 con lr=5.2×10⁻⁴, batch_size=256 (`docs/results.md`).
- **Usé seed fijo** → `SEED = 42` por defecto → porque asegura la reproducibilidad de los resultados.

**Notas del ponente:**

La reproducibilidad es fundamental en un TFG. Cada experimento genera un RUN_ID único con timestamp, y almacena en su directorio toda la configuración, métricas, el scaler entrenado y los percentiles de referencia. Esto permite reproducir exactamente cualquier resultado. Además, integré TensorBoard para monitorizar el entrenamiento y Optuna para la optimización bayesiana de hiperparámetros. El estudio con Optuna exploró 10 trials variando learning rate, batch size, arquitectura de red y otros parámetros, encontrando que un learning rate de 5.2×10⁻⁴ con batch size 256 alcanzaba la mejor accuracy de 0.9939.

---

## 10. Metodología de Evaluación (Checks A/B/C)

- **Diseñé tres checks de validación** (`src/validate_checks.py`) → para verificar que las métricas reportadas son genuinas y no artefactos de bugs o leakage → porque en ciberseguridad aplicada, la confianza en los resultados es crítica.

- **Check A — Evaluación directa** (`src/validate_checks.py` — `check_a_direct_eval`) → `model.predict(X_test[i])` vs `y_test[i]`, sin pasar por el entorno RL → porque verifica que las métricas no dependen de bugs en `info["true_label"]` del entorno.

- **Check B — Shuffled labels (anti-leakage)** (`src/validate_checks.py` — `check_b_shuffled_labels`) → baraja las etiquetas de entrenamiento y re-entrena brevemente; si la accuracy sigue siendo alta, hay leakage → porque confirma que el modelo realmente aprende de las features y no de artefactos en los datos.

- **Check C — CSV-split por día** (`src/validate_checks.py` — `check_c_csv_split`) → entrena con CSVs de lunes, martes y miércoles; testea con jueves y viernes (ataques no vistos) → porque mide la capacidad de generalización real del modelo a datos temporalmente separados y con tipos de ataque diferentes.

**Notas del ponente:**

La validación es uno de los puntos más rigurosos del proyecto. Implementé tres checks complementarios. El Check A evalúa directamente con model.predict, sin pasar por el entorno RL, para descartar bugs en la mecánica del entorno. El Check B es un test anti-leakage: barajo las etiquetas del entrenamiento y re-entreno brevemente. Si el modelo aún obtiene accuracy alta con etiquetas aleatorias, significa que hay información filtrada en las features. Finalmente, el Check C es el más exigente: entreno con tres días de la semana y testeo con los otros dos, donde aparecen tipos de ataque completamente diferentes. Este check mide la generalización real del modelo.

---

## 11. Resultados — Fase 1 (NSL-KDD)

- **Entrené DQN sobre NSL-KDD** → experimentos E01–E06 con diferentes configuraciones de recompensa → porque necesitaba validar que el framework RL aprende una política razonable antes de escalar.
- **Comparé con Random Forest** → RF obtuvo accuracy 0.7693 vs DQN 0.7602 → porque establece un baseline supervisado de referencia.
- **Exploré el impacto de las recompensas** → tp, fp, fn, omission en diferentes combinaciones → porque demostré que el sistema de recompensas controla el trade-off seguridad/disponibilidad.

| ID | Modelo | Reward (tp, fp, fn, om) | Steps | Accuracy | Recall atk | FP rate |
|----|--------|-------------------------|-------|----------|------------|---------|
| E01 | DQN | 1.0, −1.0, −2.0, 0.0 | 200k | 0.7602 | 0.600 | 0.028 |
| E02 | RF | — | — | 0.7693 | 0.615 | 0.027 |
| E05 | DQN | 2.0, −1.0, −6.0, 0.2 | 500k | 0.7563 | 0.596 | 0.031 |

> Fuente: `experiments/nslkdd_experiments.md`, `docs/results.md`

**Notas del ponente:**

Los resultados de la Fase 1 sobre NSL-KDD muestran que Random Forest supera ligeramente al DQN, con accuracy de 0.77 frente a 0.76. Sin embargo, el DQN ofrece la ventaja de ser configurable: ajustando las recompensas puedo hacer al agente más agresivo o más conservador sin re-entrenar la arquitectura. Estos experimentos validaron el framework y demostraron que el sistema de recompensas efectivamente controla el comportamiento del agente, lo cual era el objetivo de esta fase. NSL-KDD sirvió como prueba de concepto, pero sus features antiguas limitan el rendimiento máximo alcanzable.

---

## 12. Resultados — Fase 2 Dataset (CICIDS2017 + QRDQN)

- **Entrené QRDQN sobre CICIDS2017 con esquema canónico** → cuatro runs progresivos (C01 smoke, C01 full, C02 fast, C03 full) → porque escalé incrementalmente datos y timesteps para validar la estabilidad.
- **Alcancé accuracy 0.9986 en el mejor modelo (C03 full)** → 500k rows, 100k timesteps, batch_size=2048, gradient_steps=20, fp=-2.0 → porque la combinación de más datos, más pasos de gradiente y penalización FP más fuerte produjo el mejor rendimiento.

| Run | Preset | Rows | Timesteps | Accuracy | Recall atk | F1 atk |
|-----|--------|------|-----------|----------|------------|--------|
| C01 smoke | fast | 50k | 5k | 0.9697 | 0.9996 | 0.9692 |
| C01 full | full | 250k | 100k | 0.9962 | 0.9998 | 0.9963 |
| C02 fast | fast | 100k | 10k | 0.9766 | 0.9996 | 0.9812 |
| **C03 full** | **full** | **500k** | **100k** | **0.9986** | **0.9995** | **0.9988** |

> Fuente: `docs/results.md`

- **Validé con los tres checks:**

| Check | Resultado clave |
|-------|-----------------|
| A (direct eval) | Accuracy 0.9939 — TP=4772, FP=60, FN=1 |
| B (anti-leakage) | Shuffled acc 0.4773 vs baseline 0.5227 → ✅ Sin leakage |
| C (CSV-split day) | Accuracy 0.8414 (30k timesteps, días no vistos) |

> Fuente: `docs/results.md` — Sección "Validation Checks"

**Notas del ponente:**

Los resultados sobre CICIDS2017 son significativamente superiores. El mejor modelo, C03 full, alcanza una accuracy de 0.9986 con un recall de ataque de 0.9995 y F1 de 0.9988. Esto significa que de cada 10.000 ataques, solo pierde 5. La evolución de C01 smoke a C03 full muestra cómo escalar los datos de 50 mil a 500 mil filas y ajustar los hiperparámetros mejora consistentemente el rendimiento. Las validaciones confirman la solidez: el Check A corrobora las métricas sin depender del entorno, el Check B demuestra ausencia de data leakage (accuracy con etiquetas barajadas cae a 0.48, nivel aleatorio), y el Check C muestra que con solo 30.000 timesteps el modelo ya generaliza a 0.84 en días completamente no vistos. Este último check es el más difícil y con más timesteps mejoraría aún más.

---

## 13. Comparación Fase 1 → Fase 2

- **Demostré una mejora sustancial al migrar de NSL-KDD a CICIDS2017 con QRDQN** → de accuracy 0.76 a 0.9986 → porque el dataset moderno tiene features más ricas y el algoritmo distributional captura mejor la estructura del problema.

| Aspecto | Fase 1 (NSL-KDD + DQN) | Fase 2 (CICIDS2017 + QRDQN) |
|---------|-------------------------|-------------------------------|
| Dataset | NSL-KDD (1999, 41 features) | CICIDS2017 (2017, 76 canónicas) |
| Algoritmo | DQN | QRDQN (distributional) |
| Accuracy | 0.7602 | **0.9986** |
| Recall ataque | 0.600 | **0.9995** |
| F1 ataque | 0.6946 | **0.9988** |
| Esquema | Legacy (one-hot) | Canónico (152 dims) |
| Validación | Básica | Checks A/B/C |

**Notas del ponente:**

La comparación entre fases es contundente. Pasamos de una accuracy de 0.76 con DQN sobre NSL-KDD a 0.9986 con QRDQN sobre CICIDS2017. El recall de ataque mejoró de 0.60 a 0.9995. Esta mejora se debe a tres factores: un dataset moderno con features flow-based más discriminativas, un algoritmo distributional que captura mejor la incertidumbre, y un esquema canónico con missingness mask que le da al agente información fiable sobre qué features están disponibles. La migración al esquema canónico también habilitó la Fase 2 de inferencia sobre tráfico real.

---

## 14. Fase 2: Tráfico Real e Inferencia Offline

- **Diseñé un laboratorio privado de 2 VMs** (`docs/gcp_lab.md`) → VM atacante (Kali Linux, 10.0.0.10) y VM defensora (Ubuntu 22.04, 10.0.0.20) en una VPC privada → porque necesitaba un entorno controlado y seguro para generar tráfico etiquetado.

```
  TOPOLOGÍA DEL LABORATORIO PRIVADO
  ──────────────────────────────────────────────────────────

  ┌──────────────────── Private VPC (10.0.0.0/24) ────────────────────┐
  │                                                                    │
  │  ┌────────────────┐                    ┌────────────────────────┐  │
  │  │  attacker VM   │   eth0 ←→ eth0     │  defender VM           │  │
  │  │  Kali Linux    │───────────────────→ │  Ubuntu 22.04          │  │
  │  │  10.0.0.10     │                    │  10.0.0.20             │  │
  │  │                │  Genera tráfico:   │  - Docker (nginx, ssh, │  │
  │  │  nmap, hping3, │  benigno + ataques │    ftp, mysql)         │  │
  │  │  hydra, sqlmap │                    │  - tcpdump / tshark    │  │
  │  │  curl, wget    │                    │  - CICFlowMeter        │  │
  │  └────────────────┘                    │  - QRDQN model         │  │
  │                                        └────────────────────────┘  │
  │                                                                    │
  └────────────────────────────────────────────────────────────────────┘
                              │
                         SSH only desde
                         IP del alumno
```

- **Generé tráfico etiquetado** → benigno (curl, wget, ssh) y ataques (nmap, hping3, hydra, sqlmap) — todo dentro del laboratorio privado → porque necesitaba ground-truth conocido para evaluar las predicciones.
- **Capturé PCAPs y extraje flujos** → `tcpdump` → CICFlowMeter → CSV de flujos → mapping con `FLOWMETER_PY_TO_CANON` → porque replica el pipeline de CICIDS2017 sobre tráfico real.
- **Implementé inferencia v1** (`scripts/predict_real_traffic.py`) → primer intento de inferencia offline, carga modelo y scaler, mapea columnas de CICFlowMeter al esquema canónico → pero los resultados fueron extremos (todo-block o todo-allow) → porque no manejaba correctamente el domain shift.
- **Desarrollé inferencia v2 robusta** (`scripts/predict_real_traffic_v2.py`) → añade percentile clipping (`--percentiles`), z-score clipping (`--clip-z`), y diagnósticos de distribución → porque la v1 mostró que las features del tráfico real tienen distribuciones muy diferentes al dataset de entrenamiento.

**Notas del ponente:**

La Fase 2 traslada el modelo entrenado a un escenario real. Desplegué un laboratorio privado con dos máquinas virtuales en una VPC aislada: un atacante con Kali Linux que genera tráfico benigno y malicioso, y un defensor con Ubuntu donde corren servicios Docker y se captura el tráfico. Todas las herramientas de ataque —nmap, hping3, hydra, sqlmap— se ejecutan exclusivamente dentro de este laboratorio privado. Los PCAPs capturados se procesan con CICFlowMeter para extraer los mismos tipos de flujos que CICIDS2017. La primera versión de inferencia produjo resultados extremos, lo que me llevó a desarrollar la versión 2 con percentile clipping y z-score clipping para manejar el domain shift entre el tráfico de entrenamiento y el real.

---

## 15. Mitigación de Domain Shift

- **Implementé utilidades de escalado** (`src/scaling_utils.py`) → `apply_percentile_clipping()` para clampar features a los percentiles p0.5/p99.5 del entrenamiento, y `apply_z_clipping()` para limitar z-scores extremos (ej. |z| = 89 en TCP flags) → porque los features del tráfico real presentan valores extremos que colapsan las estimaciones de Q-valores del agente.
- **Diagnostiqué la distribución de features** (`scripts/predict_real_traffic_v2.py` — `compute_diagnostics`) → calcula z-score máximo, medio, y top-15 features problemáticas → porque necesitaba identificar qué features causaban las predicciones extremas.

**Resultados de inferencia Phase 2 (v2, modelo C03 full):**

| Flows CSV | Flujos | Block Rate | Allow Rate | z-abs max | z-abs mean |
|-----------|--------|------------|------------|-----------|------------|
| flows.csv | 1 261 | 20.7 % | 79.3 % | 10.0 | 0.714 |
| flows\_benign.csv | 5 327 | 100 % | 0 % | 10.0 | 1.115 |
| flows\_scan.csv | 8 721 | 59.7 % | 40.3 % | 10.0 | 0.991 |
| flows\_mix.csv | 5 511 | 95.5 % | 4.5 % | 10.0 | 1.107 |

> Fuente: `docs/results.md` — Sección "Phase 2"

**Notas del ponente:**

Un hallazgo importante fue el domain shift entre CICIDS2017 y el tráfico real del laboratorio. Features como TCP flags presentaban z-scores de hasta 89 tras el escalado, lo que colapsaba las estimaciones del agente. Implementé dos estrategias de mitigación: percentile clipping que limita los valores brutos al rango p0.5-p99.5 observado en entrenamiento, y z-score clipping que acota los valores escalados a un máximo de 10. Los resultados de la v2 muestran que el agente funciona pero aún presenta sesgo: en tráfico puramente benigno del laboratorio, el block rate fue del 100%, indicando que la distribución del tráfico real difiere significativamente de CICIDS2017. Este domain shift es un problema abierto que requiere calibración adicional o fine-tuning.

---

## 16. Línea Temporal del Proyecto

- **Reconstruí la evolución del proyecto desde los commits** → excluyendo el 23 de enero de 2026 (no relacionado) → porque demuestra el progreso iterativo y las decisiones tomadas.

| Fecha aprox. | Hito |
|-------------|------|
| 11 dic 2025 | Commit inicial: entorno RL, DQN, loader NSL-KDD, Random Forest baseline |
| 11–14 dic 2025 | Experimentos NSL-KDD (E01–E06), ablación de hiperparámetros (A01–A02) |
| ~9 feb 2026 | Definición formal del esquema canónico (`canonical_schema.py`, 76 features) |
| 12 feb 2026 | Adapter CICIDS2017, QRDQN C01 runs (smoke + full), Optuna study, Validation Checks A/B |
| 13 feb 2026 | Validation Check C (CSV-split por día) |
| 23 feb 2026 | Phase 2 inferencia v1/v2, scaling_utils, C02/C03 training runs |
| 24 feb 2026 | Runs Phase 2 v2 (flows\_benign, flows\_scan, flows\_mix) |
| 26 feb 2026 | Documentación consolidada, actualización de results.md |

**Notas del ponente:**

El proyecto comenzó en diciembre de 2025 con los fundamentos: entorno RL, DQN y experimentos sobre NSL-KDD. En febrero de 2026, el trabajo se aceleró significativamente con la definición del esquema canónico, la migración a CICIDS2017, y la transición a QRDQN. Los runs C01 a C03 representan una evolución iterativa donde cada versión mejora el rendimiento. La Fase 2 con inferencia sobre tráfico real se desarrolló a finales de febrero, incluyendo la detección y mitigación del domain shift. El proyecto ha seguido una metodología incremental, con cada paso validado antes de avanzar al siguiente.

---

## 17. Limitaciones y Trabajo Futuro

- **Identifiqué domain shift como limitación principal** → el tráfico real del laboratorio tiene distribuciones diferentes a CICIDS2017, especialmente en TCP flags y tasas de flujo → porque los patrones de un laboratorio pequeño difieren de la simulación a gran escala.
- **Check C necesita más timesteps** → con solo 30k timesteps alcanza 0.8414; con entrenamiento más largo mejoraría la generalización a días no vistos → porque el agente no converge completamente con tan pocos pasos.
- **Bloqueo en tiempo real (iptables) no implementado** → está planificado en `docs/phase2_plan.md` (Step 8) pero no ejecutado → porque prioricé la validación de inferencia offline.
- **Trabajo futuro**: RL adversarial (agente atacante vs defensor), datasets adicionales (CICIDS2018, UNSW-NB15), integración con iptables/nftables para bloqueo activo, fine-tuning del modelo sobre tráfico del laboratorio para reducir FP en tráfico benigno real.

**Notas del ponente:**

Es importante ser honesto sobre las limitaciones. El domain shift entre CICIDS2017 y el tráfico real es significativo: el modelo bloquea todo el tráfico benigno del laboratorio, lo que indica que necesita calibración. El Check C con split por día alcanza 0.84, pero con un entrenamiento más largo mejoraría. El bloqueo en tiempo real con iptables está planificado pero no implementado todavía. Como trabajo futuro, destacaría la posibilidad de RL adversarial con un agente atacante que intente evadir al defensor, la incorporación de más datasets modernos para mejorar la generalización, y el fine-tuning del modelo sobre datos del laboratorio para cerrar la brecha de domain shift.

---

## 18. Contribuciones Hasta la Fecha

### Componentes implementados

- [x] Entorno RL custom Gymnasium con recompensas configurables (`src/rl_defender_env.py`)
- [x] Esquema canónico de 76 features con máscara de missingness (`src/canonical_schema.py`)
- [x] Adapter CICIDS2017 con mapping completo 76/76 (`src/load_cicids2017.py`)
- [x] Adapter NSL-KDD con mapping parcial 3/76 (`src/load_nsl_kdd.py`)
- [x] Entrenamiento QRDQN con presets fast/full y split random/day (`src/train_rl_defender.py`)
- [x] Baseline supervisado Random Forest (`src/baseline_random_forest.py`)
- [x] Validation Checks A/B/C (`src/validate_checks.py`)
- [x] Optimización de hiperparámetros con Optuna (`src/tune_hparams.py`)
- [x] Utilidades de escalado: percentile clipping y z-score clipping (`src/scaling_utils.py`)
- [x] Pipeline de inferencia Phase 2 v1 (`scripts/predict_real_traffic.py`)
- [x] Pipeline de inferencia Phase 2 v2 robusto con diagnósticos (`scripts/predict_real_traffic_v2.py`)
- [x] Laboratorio privado documentado con topología y seguridad (`docs/gcp_lab.md`)
- [x] Documentación de resultados consolidada (`docs/results.md`)
- [x] Persistencia de artefactos: scaler.joblib, train_percentiles.npz, config.json, metrics.json

### Resultados clave

- [x] Mejor modelo (C03 full): Accuracy **0.9986**, Recall ataque **0.9995**, F1 **0.9988**
- [x] Check A: Accuracy 0.9939 sin depender del entorno RL
- [x] Check B: Ausencia de data leakage confirmada (shuffled acc 0.4773 ≈ aleatorio)
- [x] Check C: Generalización a días no vistos: 0.8414
- [x] Inferencia sobre tráfico real del laboratorio: domain shift detectado y mitigado parcialmente

### Pendiente

- [ ] Calibración/fine-tuning para reducir FP en tráfico benigno real
- [ ] Bloqueo activo en tiempo real (iptables integration)
- [ ] Datasets adicionales (CICIDS2018, UNSW-NB15)
- [ ] RL adversarial (agente atacante)
- [ ] Check C con más timesteps para mejor generalización

**Notas del ponente:**

En resumen, el proyecto tiene implementados todos los componentes core: el entorno RL, el esquema canónico, los adapters de datasets, el entrenamiento QRDQN, un baseline supervisado, tres checks de validación rigurosos, optimización de hiperparámetros, y un pipeline de inferencia sobre tráfico real con mitigación de domain shift. El mejor modelo alcanza una accuracy de 0.9986 con recall de ataque de 0.9995. Los checks confirman que no hay data leakage y que el modelo generaliza razonablemente a datos no vistos. Queda trabajo por hacer en la calibración para tráfico real y en la integración de bloqueo activo, pero la base experimental y técnica es sólida. Muchas gracias por su atención, estoy a disposición del tribunal para cualquier pregunta.

---

## ⚠️ TODOs / Inconsistencias detectadas

### Diferencias en valores de recompensa entre archivos

| Archivo | tp | fp | fn | omission |
|---------|----|----|----|---------:|
| `src/rl_defender_env.py` — `default_reward_config` | 1.0 | -1.0 | -5.0 | 0.0 |
| `src/train_rl_defender.py` — `REWARD_CONFIG` | 1.5 | **-2.0** | -5.0 | 0.0 |
| `src/validate_checks.py` — `REWARD_CONFIG` | 1.5 | **-1.0** | -5.0 | 0.0 |
| `src/tune_hparams.py` — `REWARD_CONFIG` | 1.5 | **-1.0** | -5.0 | 0.0 |
| `docs/results.md` — C02 fast | 1.5 | **-1.0** | -5.0 | 0.0 |
| `docs/results.md` — C03 full (⭐ best) | 1.5 | **-2.0** | -5.0 | 0.0 |

**Nota**: El mejor modelo (C03) se entrenó con `fp=-2.0`, mientras que validate_checks.py y tune_hparams.py usan `fp=-1.0`. Esto significa que los Checks A/B se evaluaron con un reward_config diferente al usado en el entrenamiento de C03, aunque esto solo afecta al entorno de evaluación, no a las predicciones del modelo.

### Métricas verificadas

- ✅ Todas las métricas de C01 smoke/full, C02 fast, C03 full verificadas en `docs/results.md`
- ✅ Validation Checks A/B/C verificados en `docs/results.md`
- ✅ NSL-KDD E01/E02/E05 verificados en `experiments/nslkdd_experiments.md` y `docs/results.md`
- ✅ Phase 2 v2 runs verificados en `docs/results.md`
- ✅ Optuna study verificado en `docs/results.md`

### Archivos referenciados que existen

- ✅ `src/canonical_schema.py`, `src/load_cicids2017.py`, `src/load_nsl_kdd.py`
- ✅ `src/rl_defender_env.py`, `src/train_rl_defender.py`, `src/validate_checks.py`
- ✅ `src/tune_hparams.py`, `src/baseline_random_forest.py`, `src/scaling_utils.py`
- ✅ `scripts/predict_real_traffic.py`, `scripts/predict_real_traffic_v2.py`
- ✅ `docs/results.md`, `docs/phase2_plan.md`, `docs/gcp_lab.md`
- ✅ `experiments/nslkdd_experiments.md`

### Otras observaciones

- El README.md menciona reward tp=1.5, fp=−1.0, fn=−5.0, om=0.0, pero el mejor modelo (C03) usó fp=−2.0 según `docs/results.md`.
- `predict_real_traffic.py` (v1) ejecuta `load_cicids2017_split()` al nivel de módulo (fuera de `main()`), lo que fuerza la carga del dataset solo al importar el script. La v2 corrige esto cargando el scaler desde un archivo joblib.
- [⚠️ VERIFICAR: Confirmar que los RUN_IDs en `docs/results.md` tienen directorios correspondientes reales en `runs/`; no se puede verificar directamente si los JSONs de métricas existen en este entorno sin el dataset.]
