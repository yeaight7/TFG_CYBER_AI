# Defensa TFG — Script de Presentación Oral

**Autor**: Javier Rivero Iglesias  
**Duración estimada**: 20 a 30 minutos  
**Nota**: Este script está alineado sección por sección con `DEFENSA_TFG_PROGRESO.md`.  

---

## 1. Introducción y Motivación (~2 min)

Buenos días, miembros del tribunal. Mi nombre es Javier Rivero Iglesias y presento mi Trabajo de Fin de Grado titulado "Agente de Ciberseguridad con Aprendizaje por Refuerzo".

El objetivo de este trabajo es diseñar e implementar un **agente defensor** capaz de decidir si un flujo de red debe ser **permitido o bloqueado**, utilizando Aprendizaje por Refuerzo. ¿Por qué RL y no simplemente un clasificador supervisado como Random Forest o una red neuronal? Porque en ciberseguridad los costes son fundamentalmente **asimétricos**: dejar pasar un ataque puede comprometer toda una infraestructura, mientras que bloquear por error un paquete legítimo es una molestia menor. El Aprendizaje por Refuerzo permite codificar directamente estos costes asimétricos en un **sistema de recompensas configurable**, de forma que puedo ajustar la postura de seguridad del agente —de conservadora a agresiva— simplemente cambiando los valores de recompensa, sin necesidad de re-entrenar la arquitectura completa.

El proyecto se estructura en dos fases: primero, entrenamiento y validación sobre datasets históricos y modernos; y después, evaluación sobre tráfico real capturado en un laboratorio privado.

Paso ahora a describir la arquitectura general del sistema.

---

## 2. Arquitectura General del Proyecto (~1.5 min)

Diseñé una arquitectura modular con un pipeline de seis etapas. Los datos de entrada —CSVs de datasets o flujos extraídos de PCAPs— pasan por un **adapter** específico del dataset que los convierte al **esquema canónico de 76 features**. A continuación, se añade una **máscara de missingness** de otras 76 dimensiones, resultando en un vector de observación de **152 dimensiones**. Este vector se normaliza con un `StandardScaler` ajustado solo en el conjunto de entrenamiento, y entra al entorno Gymnasium custom donde el agente QRDQN observa el flujo y decide PERMIT o BLOCK.

La clave del diseño es que cada componente es **independiente y reemplazable**: puedo cambiar el dataset añadiendo un nuevo adapter, cambiar el algoritmo sustituyendo QRDQN por otro, o cambiar la estrategia de escalado, todo sin afectar al resto del pipeline. El código está organizado en `src/` con módulos separados: `canonical_schema.py` para el esquema de features, `load_cicids2017.py` y `load_nsl_kdd.py` como adapters, `rl_defender_env.py` para el entorno, `train_rl_defender.py` para el entrenamiento, y `validate_checks.py` para la validación.

---

## 3. Datasets: NSL-KDD y CICIDS2017 (~1.5 min)

Trabajo con dos datasets principales. **NSL-KDD** es un benchmark histórico derivado de KDD Cup 1999 con 41 features basadas en conexiones TCP. Lo usé como prueba de concepto en la Fase 1 para validar que el framework RL aprende una política razonable. Sin embargo, NSL-KDD tiene limitaciones fundamentales: sus features son antiguas, basadas en conexiones y no en flujos modernos, y no son extraíbles de PCAPs con herramientas actuales. Por esta razón, **NSL-KDD no forma parte del modelo final**.

**CICIDS2017** es el dataset principal: generado por el Canadian Institute for Cybersecurity en 2017, contiene aproximadamente 2.8 millones de flujos distribuidos en 5 días de tráfico —de lunes a viernes—, con unas 80 columnas de features extraídas por CICFlowMeter directamente de PCAPs. Incluye ataques modernos como DDoS, Port Scan, Brute Force, Web Attacks y Botnets. Este dataset **define el esquema canónico** porque sus features son exactamente las que un flow extractor produce al procesar tráfico real, lo cual es esencial para la transición a la Fase 2.

El sistema está diseñado para soportar **N datasets** mediante adapters independientes: cada loader mapea las columnas del dataset al esquema canónico, con la posibilidad de añadir datasets futuros como CICIDS2018 o UNSW-NB15.

---

## 4. Ingeniería de Features y Esquema Canónico (~1.5 min)

El esquema canónico es una de las decisiones de diseño más importantes del proyecto, definido formalmente en `src/canonical_schema.py` como la lista `FEATURES_CANON`. Son **76 features** flow-based, organizadas en categorías: estadísticas generales del flujo (duración, paquetes, bytes), estadísticas de tamaño de paquete forward y backward, tasas de flujo (bytes/s, packets/s), inter-arrival times del flujo y por dirección, **8 flags TCP** (SYN, ACK, FIN, RST, PSH, URG, ECE, CWE), longitud de cabeceras, ratios derivados, bulk statistics, subflow statistics, ventanas TCP iniciales, y tiempos activos e idle.

El criterio de selección fue estricto. Todas las features debían cumplir cinco condiciones: existir en CICIDS2017, ser extraíbles de tráfico real mediante CICFlowMeter, no causar data leakage, ser numéricas, y ser estables entre datasets. Las columnas **excluidas explícitamente** —implementado en `_drop_identifier_like_columns` de `load_cicids2017.py`— son: Flow ID, Timestamp, Source IP, Destination IP, Source Port y Destination Port. Destination Port, por ejemplo, se elimina porque ciertos ataques apuntan a puertos específicos, y si el modelo lo ve, aprende un atajo en vez de patrones de flujo genuinos.

Los nombres de columnas se normalizaron de Title Case con espacios a `lower_snake_case`. El mapping `CICIDS2017_TO_CANON` cubre las 76 features completas; el mapping `NSL_KDD_TO_CANON` solo cubre 3 de 76 —`duration`, `src_bytes`, `dst_bytes`— lo que ilustra la incompatibilidad entre datasets antiguos y modernos.

---

## 5. Máscara de Missingness (~1 min)

Para manejar la heterogeneidad entre datasets, implementé una **máscara de missingness binaria** de 76 dimensiones. Para cada feature canónica, `m_i = 1` indica que estaba presente y era un valor válido en el dataset original, mientras que `m_i = 0` indica que fue imputada. La imputación por defecto es con valor cero, definido como `DEFAULT_IMPUTATION_VALUE = 0.0` en `canonical_schema.py`, lo cual es semánticamente coherente: para contadores, bytes y tasas, cero significa ausencia de actividad.

El vector de observación final es la concatenación de features y máscara: `obs = [x_1...x_76, m_1...m_76]`, resultando en `NUM_OBSERVATION_FEATURES = 152` dimensiones. Para CICIDS2017, la máscara es prácticamente todo unos porque mapea las 76 features. Para NSL-KDD, solo 3 features están presentes; las restantes 73 tienen máscara en cero, lo que le indica al agente que esas posiciones no son fiables. Esta arquitectura permite que en el futuro se entrene con múltiples datasets simultáneamente, manteniendo siempre la misma dimensión del espacio de observación.

---

## 6. Pipeline de Preprocesamiento (~1 min)

El preprocesamiento sigue un pipeline riguroso implementado en `load_cicids2017.py`. Primero, **coerción numérica** de todas las columnas con `pd.to_numeric(errors="coerce")`. Después, reemplazo de infinitos por NaN y relleno con cero. Luego, el `StandardScaler` se ajusta **exclusivamente** en el conjunto de entrenamiento con `scaler.fit_transform(X_train)` y se aplica al test con `scaler.transform(X_test)`, para evitar data leakage temporal. Este scaler se persiste como `scaler.joblib` para usarlo después en inferencia Phase 2.

Además, calculo y persisto los **percentiles p0.5 y p99.5** del entrenamiento como `train_percentiles.npz`, que sirven para el percentile clipping en Phase 2. Implementé dos modos de split: **random** (split estratificado 80/20, útil para evaluación rápida) y **day** (entreno con lunes, martes y miércoles, testeo con jueves y viernes), que es más realista porque los días de test contienen tipos de ataque no vistos en entrenamiento.

---

## 7. Diseño del Entorno RL (~1.5 min)

El entorno Gymnasium custom, `RLDatasetDefenderEnv` en `rl_defender_env.py`, encapsula el dataset como un entorno interactivo. En cada paso, el agente observa un vector de 152 dimensiones correspondiente a un flujo de red, toma una acción —`0 = PERMIT` o `1 = BLOCK`— y recibe una recompensa basada en la matriz de confusión.

El **sistema de recompensas** es la pieza clave. Los valores por defecto del entorno son: tp=1.0, fp=-1.0, fn=-5.0, omission=0.0. Pero en el entrenamiento del mejor modelo C03, usé valores ajustados: tp=1.5, fp=-2.0, fn=-5.0, omission=0.0. La penalización por falso negativo es cinco veces mayor que por falso positivo, lo que codifica la asimetría: es peor dejar pasar un ataque que bloquear tráfico legítimo. El modelo de episodio trata cada flujo como un paso independiente, y el episodio termina cuando se agotan las muestras o se alcanza `max_steps_per_episode`.

Un aspecto del diseño que me parece especialmente elegante es que **puedo cambiar la postura de seguridad del agente solo modificando los pesos de recompensa**, sin tocar ni la arquitectura de la red ni el código del entorno. Esto es una ventaja directa del enfoque RL sobre el supervisado.

---

## 8. Algoritmos: DQN → QRDQN (~1.5 min)

La evolución algorítmica fue de **DQN a QRDQN**. En la Fase 1, usé DQN estándar —Deep Q-Network, el algoritmo que combina Q-learning con redes neuronales profundas para estimar Q(s,a)— sobre NSL-KDD. Alcancé una accuracy de 0.76, comparable al Random Forest baseline que obtiene 0.77. Estos resultados validaron que el framework funciona, pero el rendimiento era limitado.

Al migrar a CICIDS2017 con el esquema canónico, transicioné a **QRDQN** —Quantile Regression DQN—, una variante distributional que en vez de estimar un único valor esperado Q(s,a), estima la **distribución completa del retorno** mediante regresión por cuantiles. Esto es particularmente relevante en ciberseguridad porque las recompensas son muy asimétricas: el agente necesita entender no solo el retorno medio sino la variabilidad del riesgo asociado a cada acción.

La implementación usa `sb3_contrib.QRDQN` con una red MLP de dos capas —512 y 256 neuronas—, learning rate de 1×10⁻⁴, batch size de 2048 en modo full, 20 gradient steps, y train frequency de 100 pasos. El gamma es 0.99 y el buffer size se dimensiona dinámicamente según los timesteps totales.

---

## 9. Entrenamiento, Tracking y Reproducibilidad (~1 min)

La reproducibilidad es fundamental en un trabajo académico. Cada experimento genera un **RUN_ID único** con formato `C03_qrdqn_cicids2017_canonical_full_random_20260223_232439`, que identifica unívocamente las condiciones del run. En el directorio `runs/<category>/<RUN_ID>/` se almacenan todos los artefactos: `config.json` con la configuración completa de hiperparámetros, `metrics.json` con las métricas de evaluación, `scaler.joblib` y `train_percentiles.npz`.

Integré **TensorBoard** para monitorizar curvas de aprendizaje durante el entrenamiento, y **Optuna** para la optimización bayesiana de hiperparámetros. El estudio con Optuna exploró 10 trials variando learning rate, batch size, gradient steps, arquitectura de red, gamma y train frequency. El mejor trial alcanzó una accuracy de 0.9939 con learning rate de 5.2×10⁻⁴ y batch size de 256, resultados almacenados en `runs/optuna/study_20260212_222134.json`. Todos los experimentos usan seed fijo (42 por defecto) para garantizar la reproducibilidad.

---

## 10. Metodología de Evaluación (~1.5 min)

Implementé **tres checks de validación** en `validate_checks.py` para verificar que las métricas son genuinas.

El **Check A** realiza una evaluación directa: llama a `model.predict(X_test[i])` y compara con `y_test[i]`, sin pasar por el entorno RL. Esto descarta posibles bugs en la mecánica del entorno, como errores en `info["true_label"]`. El resultado fue una accuracy de 0.9939, con TP=4772, FP=60, FN=1, y TN=5167.

El **Check B** es un test anti-leakage: barajo aleatoriamente las etiquetas de entrenamiento y re-entreno brevemente durante 2000 timesteps. Si el modelo aún obtiene accuracy alta con etiquetas aleatorias, significa que las features contienen información filtrada de la etiqueta. El resultado fue una accuracy con etiquetas barajadas de 0.4773, por debajo del baseline de clase mayoritaria de 0.5227. Esto confirma inequívocamente que **no hay data leakage**.

El **Check C** es el más exigente: entreno con los CSVs de lunes, martes y miércoles, y testeo con los de jueves y viernes, donde aparecen tipos de ataque completamente diferentes. Con 30.000 timesteps, el modelo alcanza una accuracy de 0.8414. Es el resultado más bajo pero también el más significativo, porque demuestra la capacidad de generalización a datos temporalmente separados y con ataques no vistos.

---

## 11. Resultados — Fase 1 (~1 min)

Los resultados de la Fase 1 sobre NSL-KDD, documentados en `experiments/nslkdd_experiments.md`, muestran que Random Forest supera ligeramente al DQN: accuracy de 0.7693 frente a 0.7602. El recall de ataque del RF fue 0.615 contra 0.600 del DQN, con FP rates similares (0.027 vs 0.028).

Sin embargo, los experimentos E01 a E06 demostraron que el **sistema de recompensas controla efectivamente el comportamiento del agente**: con recompensas más agresivas (E05, tp=2.0, fn=-6.0), el recall sube a 0.596 pero el FP rate aumenta a 0.031. Sin recompensa por omisión (E06, om=0.0), los resultados son similares a E05, indicando que las penalizaciones son más determinantes que los bonuses. Estos resultados validaron el framework y establecieron la base para escalar a CICIDS2017.

---

## 12. Resultados — CICIDS2017 + QRDQN (~1.5 min)

Los resultados sobre CICIDS2017 representan un salto cualitativo. El mejor modelo, **C03 full**, entrenado con 500.000 filas durante 100.000 timesteps, alcanza una **accuracy de 0.9986**, con recall de ataque de **0.9995** y F1 de **0.9988**. Esto significa que de cada 10.000 ataques, el modelo solo deja pasar 5.

La evolución fue progresiva: C01 smoke con 50k filas y 5k timesteps ya alcanzaba 0.97. C01 full con 250k filas y 100k timesteps subió a 0.996. C02 fast validó la estabilidad con 100k filas. Y C03 full, con 500k filas, gradient_steps=20 y fp=-2.0, alcanzó el máximo de 0.9986.

Las validaciones confirman la solidez: el Check A corrobora una accuracy de 0.9939 sin depender del entorno; con solo 1 falso negativo sobre 4773 ataques en la muestra. El Check B confirma ausencia de leakage con una accuracy barajada de 0.4773. Y el Check C demuestra generalización a días no vistos con 0.8414, un resultado prometedor considerando que solo usó 30.000 timesteps.

---

## 13. Comparación Fase 1 → Fase 2 (~0.5 min)

La comparación entre fases es contundente. Pasamos de accuracy 0.76 con DQN sobre NSL-KDD a 0.9986 con QRDQN sobre CICIDS2017. El recall de ataque mejoró de 0.60 a 0.9995. Esta mejora se debe a tres factores convergentes: un dataset moderno con features flow-based más discriminativas, un algoritmo distributional que captura mejor la incertidumbre del retorno, y un esquema canónico con missingness mask que proporciona al agente información fiable sobre la calidad de las features.

---

## 14. Fase 2: Tráfico Real e Inferencia Offline (~2 min)

La Fase 2 traslada el modelo entrenado a un escenario con tráfico real. Diseñé y documenté un **laboratorio privado** con dos máquinas virtuales en una VPC aislada: una VM atacante con Kali Linux en 10.0.0.10 y una VM defensora con Ubuntu 22.04 en 10.0.0.20. Las reglas de seguridad son estrictas: VPC privada sin conectividad externa, SSH solo desde mi IP, firewall deny-all para egress, y todas las herramientas de ataque —nmap, hping3, hydra, sqlmap— se ejecutan exclusivamente dentro de este laboratorio privado.

En la VM defensora corrí servicios Docker (nginx, SSH, FTP, MySQL) como targets, y capturé el tráfico con tcpdump. Luego extraje flujos con CICFlowMeter, obteniendo CSVs con las mismas columnas que CICIDS2017. El mapping de CICFlowMeter Python a nombres canónicos está en `FLOWMETER_PY_TO_CANON`, que mapea los nombres con formato diferente —por ejemplo, `tot_fwd_pkts` → `total_fwd_packets`— al esquema canónico.

La primera versión del script de inferencia (`predict_real_traffic.py`) produjo resultados extremos: todo-block o todo-allow. Esto me llevó a desarrollar la **versión 2** (`predict_real_traffic_v2.py`), un pipeline robusto que añade percentile clipping sobre features brutas, z-score clipping sobre features escaladas, y diagnósticos detallados de distribución para detectar domain shift.

Los resultados de la v2 muestran que el agente funciona pero con limitaciones: sobre tráfico general del laboratorio, el block rate fue del 20.7%. Pero sobre tráfico puramente benigno, el block rate fue del 100%, indicando un **domain shift significativo** entre CICIDS2017 y el tráfico real del laboratorio. Esto es un hallazgo importante que motiva trabajo futuro de calibración.

---

## 15. Mitigación de Domain Shift (~1 min)

Para abordar el domain shift, implementé dos estrategias complementarias en `scaling_utils.py`. La función `apply_percentile_clipping()` limita cada feature bruta al rango de los percentiles p0.5 y p99.5 calculados sobre el entrenamiento, antes de aplicar el scaler. Esto previene que valores extremos del tráfico real —por ejemplo, TCP flag counts mucho más altos que en CICIDS2017— distorsionen el escalado.

La función `apply_z_clipping()` actúa después del escalado, limitando los z-scores a un máximo absoluto de 10.0. Sin este clipping, observamos z-scores de hasta 89 en features de TCP flags, lo cual colapsaba las estimaciones de Q-valores del agente. El script v2 incluye además una función de diagnóstico que reporta las top-15 features por z-score máximo, lo que permitió identificar exactamente dónde se producía el shift.

---

## 16. Línea Temporal del Proyecto (~0.5 min)

El proyecto comenzó en **diciembre de 2025** con los fundamentos: entorno RL, DQN y experimentos sobre NSL-KDD. En **febrero de 2026** se aceleró con la definición del esquema canónico, la migración a CICIDS2017, y la transición a QRDQN. Los runs C01 a C03 muestran la mejora iterativa. La Fase 2 con inferencia sobre tráfico real se desarrolló a finales de febrero, incluyendo la detección y mitigación del domain shift. Todo el progreso está documentado en `docs/results.md` con métricas extraídas directamente de los JSONs de cada run.

---

## 17. Limitaciones y Trabajo Futuro (~1 min)

Es importante ser transparente sobre las limitaciones. El **domain shift** entre CICIDS2017 y el tráfico del laboratorio es significativo: el modelo bloquea el 100% del tráfico benigno real, lo que indica la necesidad de calibración o fine-tuning sobre datos del laboratorio. El **Check C** con split por día alcanza 0.84 con solo 30.000 timesteps; con un entrenamiento más largo mejoraría. El **bloqueo en tiempo real** con iptables está planificado y documentado en `docs/phase2_plan.md`, pero no implementado aún.

Como trabajo futuro, destacaría cuatro líneas principales: primero, **calibración para tráfico real** mediante fine-tuning o transfer learning sobre datos del laboratorio. Segundo, **RL adversarial** con un agente atacante que intente evadir al defensor, creando un juego de suma cero. Tercero, **más datasets** como CICIDS2018 y UNSW-NB15 para mejorar la generalización. Y cuarto, la **integración con iptables/nftables** para bloqueo activo en tiempo real.

---

## 18. Conclusión y Cierre (~1 min)

En resumen, este TFG demuestra la viabilidad del Aprendizaje por Refuerzo para la ciberseguridad defensiva. El sistema implementado incluye: un esquema canónico de 76 features con máscara de missingness para generalización multi-dataset; un entorno Gymnasium custom con recompensas configurables; un agente QRDQN que alcanza **accuracy de 0.9986** y **recall de ataque de 0.9995** sobre CICIDS2017; tres checks de validación rigurosos que confirman ausencia de data leakage y capacidad de generalización; y un pipeline de inferencia sobre tráfico real con mitigación de domain shift.

El código es modular, reproducible y extensible. Cada componente —loaders, esquema canónico, entorno, entrenamiento, validación— funciona de forma independiente y está documentado. Los artefactos de cada run se persisten para garantizar la trazabilidad total.

Queda trabajo por hacer: la calibración para tráfico real, el bloqueo activo, y la extensión a escenarios adversariales. Pero la base experimental y técnica es sólida, y el enfoque RL ha demostrado ventajas claras sobre métodos supervisados en términos de configurabilidad y adaptabilidad del sistema de recompensas.

Muchas gracias por su atención. Estoy a disposición del tribunal para cualquier pregunta.
