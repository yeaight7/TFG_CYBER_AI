# Documentación de Experimentos

Esta carpeta recopila la documentación detallada de todos los experimentos realizados en el TFG. El objetivo es mantener un registro sistemático de las configuraciones probadas, resultados obtenidos y conclusiones extraídas para facilitar la reproducibilidad y el análisis comparativo.

## 📋 Contenido

### Experimentos por Dataset

- **`nslkdd_experiments.md`**: Experimentos sobre el dataset NSL-KDD (Phase 1)
  - Agente RL basado en DQN (entorno `RLDatasetDefenderEnv`)
  - Comparativa con baselines supervisados (Random Forest)
  - Análisis de diferentes configuraciones de recompensas
  - Evaluación de hiperparámetros

### Resultados Consolidados

- **[`docs/results.md`](../docs/results.md)**: Métricas consolidadas de TODOS los experimentos
  - Runs de entrenamiento CICIDS2017 (C01, C02, C03)
  - Validation Checks (A, B, C)
  - Optuna hyperparameter study
  - Phase 2 inference runs (lab-captured traffic)

### Futuros Experimentos (Planificados)

- **`cross_dataset.md`**: Evaluación de generalización entre datasets
- **`algorithms_comparison.md`**: Comparativa exhaustiva de algoritmos RL (DQN, PPO, A2C, SAC)

## 🏷️ Convención de IDs de Experimento

Para mantener un registro organizado, cada experimento tiene un ID único siguiendo esta nomenclatura:

### Prefijos por Dataset
- **`E01`, `E02`, ...**: Experimentos con NSL-KDD (Phase 1)
- **`C01`, `C02`, ...**: Experimentos con CICIDS2017 (Phase 1 — dataset principal)
- **`P2_*`, `P2v2_*`**: Runs de inferencia Phase 2 (lab-captured traffic)
- **`VAL_*`**: Validation checks (A, B, C)
- **`study_*`**: Optuna hyperparameter studies
- **`X01`, `X02`, ...**: Experimentos cross-dataset (futuro)

### Sufijos Opcionales (para variantes)
- **`E01a`, `E01b`, ...**: Variantes del mismo experimento con cambios menores
- **`E01-RF`**: Baseline de Random Forest para experimento E01
- **`E01-PPO`**: Experimento E01 replicado con PPO en lugar de DQN

### Ejemplos
- `E01`: Primer experimento baseline con DQN sobre NSL-KDD 20%
- `E02`: Baseline supervisado Random Forest sobre NSL-KDD 20%
- `E03`: DQN con recompensas ajustadas (penalización fuerte en FN)
- `E04`: Mismo experimento E03 pero con dataset completo

## 📊 Estructura de Documentación de Experimentos

Cada archivo de experimentos sigue una estructura estándar:

### 1. Tabla Resumen
- ID del experimento
- Modelo/algoritmo utilizado
- Configuración del dataset
- Hiperparámetros clave
- Métricas de evaluación
- Observaciones principales

### 2. Detalles por Experimento
Para experimentos destacados, se incluye:
- **Motivación**: ¿Por qué se realizó este experimento?
- **Configuración completa**: Todos los hiperparámetros
- **Resultados detallados**: Métricas, confusion matrix, curvas de aprendizaje
- **Análisis**: Interpretación de resultados
- **Conclusiones**: Lecciones aprendidas y siguientes pasos

### 3. Comparativas
- Gráficos comparativos entre experimentos
- Análisis de trade-offs (FP vs FN, accuracy vs recall)
- Recomendaciones sobre qué configuración usar según el caso de uso

## 🎯 Objetivos de los Experimentos

### Fase 1: Baseline y Proof of Concept
✅ **Completado**: Experimentos E01-E06 (NSL-KDD), C01-C03 (CICIDS2017)
- Establecer baseline de RL (DQN/QRDQN) y supervisado (RF)
- Explorar diferentes configuraciones de recompensas
- Validar que el agente RL puede aprender políticas efectivas
- Mejor modelo: C03 full (QRDQN, 500k rows, accuracy 0.9986, F1 attack 0.9988)

### Fase 2: Optimización de Hiperparámetros
✅ **Completado**: Optuna study con 10 trials
- Grid search de learning_rate, batch_size, gradient_steps, gamma, train_freq
- Mejor resultado: accuracy 0.9939 (lr=5.2e-4, batch=256, grad_steps=10)
- Resultados en `runs/optuna/study_20260212_222134.json`

### Fase 3: Validación y Anti-Leakage
✅ **Completado**: Checks A, B, C
- Check A: Evaluación directa sin entorno → accuracy 0.9939
- Check B: Labels barajados → accuracy 0.4773 (confirma NO leakage)
- Check C: Split por día (Mon-Wed train, Thu-Fri test) → accuracy 0.8414

### Fase 4: Inferencia sobre Tráfico Real (Phase 2)
🔶 **En progreso**: Runs P2 y P2v2
- Inferencia offline sobre PCAPs capturados en lab privado
- Pipeline v2 con z-score clipping para robustez ante distribución shift
- Resultados preliminares muestran necesidad de calibración/fine-tuning

### Fase 5: Comparativa de Algoritmos RL
📅 **Planificado**:
- DQN vs PPO vs A2C vs SAC
- On-policy vs Off-policy en este dominio
- Análisis de sample efficiency

### Fase 6: Adversarial Robustness
📅 **Planificado**:
- Evaluación contra evasion attacks
- Adversarial training
- Certified robustness

## 📈 Métricas de Evaluación

En todos los experimentos se reportan las siguientes métricas:

### Métricas de Clasificación
- **Accuracy**: Proporción de decisiones correctas
- **Precision (clase ataque)**: TP / (TP + FP) - Qué tan confiables son los bloqueos
- **Recall (clase ataque)**: TP / (TP + FN) - Qué proporción de ataques se detecta
- **F1-Score**: Media armónica de precision y recall
- **FP Rate**: FP / (FP + TN) - Proporción de tráfico legítimo bloqueado

### Métricas RL-Específicas
- **Reward acumulada**: Suma de recompensas por episodio
- **Steps por episodio**: Cuántas muestras procesa antes de terminar
- **Convergencia**: Número de timesteps hasta estabilización

### Métricas de Eficiencia
- **Tiempo de entrenamiento**: Wall-clock time
- **Memoria utilizada**: RAM/VRAM peak
- **Tiempo de inferencia**: Latencia por predicción

## 🔄 Proceso de Experimentación

### 1. Planificación
- Definir hipótesis a validar
- Elegir configuración base
- Determinar métricas clave

### 2. Ejecución
```bash
cd src
# Configurar en train_rl_defender.py:
# - REWARD_CONFIG
# - use_20_percent
# - total_timesteps
# - hiperparámetros del modelo
python train_rl_defender.py
```

### 3. Documentación
- Registrar configuración completa en tabla
- Copiar resultados (confusion matrix, classification report)
- Guardar modelo en `models/` con nombre descriptivo
- Añadir observaciones y análisis

### 4. Análisis Comparativo
- Comparar con experimentos previos
- Identificar mejoras o degradaciones
- Formular nuevas hipótesis

## 🛠️ Herramientas de Análisis

### Scripts de Análisis (Futuro)
- `analyze_experiments.py`: Genera gráficos comparativos automáticos
- `best_model_selector.py`: Selecciona mejor modelo según métricas objetivo
- `hyperparameter_viz.py`: Visualiza impacto de hiperparámetros

### Integración con Experiment Tracking
Se recomienda integrar con herramientas de tracking:
- **MLflow**: Para tracking automático de experimentos
- **Weights & Biases (W&B)**: Para visualización en tiempo real
- **TensorBoard**: Para monitorizar curvas de aprendizaje

Ejemplo con TensorBoard:
```python
from stable_baselines3.common.callbacks import TensorboardCallback

model.learn(
    total_timesteps=1_000_000,
    callback=TensorboardCallback(),
    tb_log_name="E07_experiment"
)
```

Visualizar:
```bash
tensorboard --logdir ./runs
```

## 📝 Plantilla para Nuevos Experimentos

Al añadir un nuevo experimento, incluir en la tabla:

```markdown
| ID   | Modelo | Dataset | Reward (tp, fp, fn, om) | Steps | Acc | Rec atk | FP rate | Notas |
|------|--------|---------|-------------------------|-------|-----|---------|---------|-------|
| Exxx | DQN    | ...     | x.x, -x.x, -x.x, x.x   | xxxk  | x.xx| x.xxx   | x.xxxx  | ...   |
```

Y opcionalmente, añadir sección detallada:
```markdown
### Experimento Exxx: [Título Descriptivo]

**Motivación**: [Por qué se realiza]

**Configuración**:
- Modelo: [DQN/PPO/etc]
- Dataset: [NSL-KDD 20%/Full/etc]
- Reward config: {...}
- Hiperparámetros: {...}

**Resultados**:
[Confusion matrix y métricas]

**Análisis**:
[Interpretación de resultados]

**Conclusiones**:
[Lecciones aprendidas]
```

## 🔗 Referencias

- Ver `nslkdd_experiments.md` para experimentos ya realizados
- Código de entrenamiento en `../src/train_rl_defender.py`
- Definición del entorno en `../src/rl_defender_env.py`
- Baseline supervisado en `../src/baseline_random_forest.py`
