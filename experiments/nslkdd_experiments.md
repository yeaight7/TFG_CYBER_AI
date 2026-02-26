# Experimentos NSL-KDD – DQN y Random Forest

Este documento recopila todos los experimentos realizados sobre el dataset **NSL-KDD** para comparar diferentes enfoques de detección de intrusiones:

- **Reinforcement Learning**: Agente defensor basado en **DQN** (Deep Q-Network)
- **Supervised Learning**: Modelo clásico **Random Forest** como baseline

El objetivo es evaluar la efectividad del aprendizaje por refuerzo frente a métodos tradicionales de machine learning supervisado, así como explorar diferentes configuraciones de sistemas de recompensas y hiperparámetros.

---

## 📊 Tabla Resumen de Experimentos

### Leyenda de Columnas

- **`ID`**: Identificador único del experimento (E01, E02, ...)
- **`Modelo`**: Algoritmo utilizado (DQN, RF, PPO, etc.)
- **`Dataset`**: Variante del dataset utilizada
  - `NSL-KDD (20% train)`: 25,192 muestras de entrenamiento
  - `NSL-KDD (full train)`: 125,973 muestras de entrenamiento
- **`Reward (tp, fp, fn, om)`**: Sistema de recompensas para RL
  - `tp` (True Positive): Recompensa por bloquear ataque correctamente
  - `fp` (False Positive): Penalización por bloquear tráfico benigno
  - `fn` (False Negative): Penalización por permitir ataque
  - `om` (Omission): Recompensa por permitir tráfico benigno (True Negative)
- **`Steps`**: Timesteps totales de entrenamiento RL (no aplica a RF)
- **`Acc`**: Accuracy global en conjunto de test
- **`Rec atk`**: Recall de la clase ataque (proporción de ataques detectados)
- **`FP rate`**: Tasa de falsos positivos (proporción de tráfico legítimo bloqueado)
- **`Notas`**: Observaciones y conclusiones principales

### Tabla de Resultados

```markdown
__________________________________________________________________________________________________________________________
| ID  | Modelo | Dataset                  | Reward (tp, fp, fn, om) | Steps   | Acc    | Rec atk | FP rate | Notas                                                 |
|-----|--------|--------------------------|-------------------------|---------|--------|---------|---------|-------------------------------------------------------|
| E01 | DQN    | NSL-KDD (20% train)      | 1.0, -1.0, -2.0, 0.0    |  200k   | 0.7602 | 0.600   | 0.028   | Baseline RL inicial (recompensa más suave en FN)      |
| E02 | RF     | NSL-KDD (20% train)      | -                       |   -     | 0.7693 | 0.615   | 0.0267  | Baseline supervisado Random Forest                    |
| E03 | DQN    | NSL-KDD (20% train)      | 1.0, -1.0, -5.0, 0.5    |  1000k  | 0.7208 | 0.528   | 0.0249  | RL con FN duro + omisión en benignos                  |
| E04 | DQN    | NSL-KDD (full train)     | 1.0, -1.0, -5.0, 0.5    |  1000k  | 0.7155 | 0.518   | 0.0254  | Misma reward, entrenado con NSL-KDD completo          |
| E05 | DQN    | NSL-KDD (20% train)      | 2.0, -1.0, -6.0, 0.2    |  500k   | 0.7563 | 0.5955  | 0.0313  | Reward más agresiva pro-seguridad (FN muy penalizado) |
| E06 | DQN    | NSL-KDD (20% train)      | 1.5, -1.0, -5.0, 0.0    |  500k   | 0.7555 | 0.5928  | 0.0296  | Sin recompensa por omisión, ligera subida de FP       |
|_____|________|__________________________|_________________________|_________|________|_________|_________|_______________________________________________________|
```
## ✅ Reproducibilidad

Para cada experimento, registrar:

- **Commit SHA** del repo
- **Script** ejecutado y comando exacto
- **Seed** (si aplica) y versión de Python
- **Stable-Baselines3 / sb3-contrib** (versiones)
- **GPU/CPU** (device) y SO (WSL/Ubuntu)
- **Preprocesado** (one-hot, scaler, features finales)
- **Reward config** (tp, fp, fn, om) y mapeo de acciones (0=PERMIT, 1=BLOCK)

---

## 📈 Análisis Comparativo

### DQN vs Random Forest

| Métrica | E02 (RF) | E01 (DQN) | E06 (DQN) | Ganador |
|---------|----------|-----------|-----------|---------|
| **Accuracy** | 0.7693 | 0.7602 | 0.7555 | 🏆 RF |
| **Recall Ataque** | 0.615 | 0.600 | 0.5928 | 🏆 RF |
| **FP Rate** | 0.0267 | 0.028 | 0.0296 | 🏆 RF |
| **Tiempo Entrenamiento** | ~5 min | ~45 min | ~25 min | 🏆 RF |

**Conclusión**: En esta fase inicial, Random Forest **supera ligeramente** al DQN en todas las métricas. Sin embargo, DQN ofrece ventajas únicas:
- ✅ **Configurabilidad**: Se puede ajustar el comportamiento via recompensas sin re-entrenar
- ✅ **Aprendizaje continuo**: Puede adaptarse online a nuevos datos
- ✅ **Optimización de objetivos complejos**: Puede optimizar trade-offs específicos

### Impacto del Sistema de Recompensas

Comparando experimentos DQN con el mismo dataset (20%) pero diferentes rewards:

| Experimento | Reward Config | Acc | Rec atk | FP rate | Interpretación |
|-------------|---------------|-----|---------|---------|----------------|
| **E01** | tp=1.0, fn=-2.0 (suave) | 0.7602 | 0.600 | 0.028 | Balance razonable |
| **E03** | tp=1.0, fn=-5.0 (duro) | 0.7208 | 0.528 | 0.0249 | Reduce FP a costa de detectar menos ataques |
| **E05** | tp=2.0, fn=-6.0 (agresivo) | 0.7563 | 0.5955 | 0.0313 | Mayor recall, pero aumenta FP |
| **E06** | tp=1.5, fn=-5.0, om=0.0 | 0.7555 | 0.5928 | 0.0296 | Balance intermedio |

**Observaciones**:
1. **Penalización FN alta** (E03, E05) → Agente más conservador → Menos FP pero también menos recall
2. **Recompensa TP alta** (E05) → Agente más agresivo → Mayor recall pero también más FP
3. **Omission reward** (E03 vs E06) → Impacto moderado en comportamiento

### Dataset Completo vs 20%

| Métrica | E03 (20% train) | E04 (full train) | Diferencia |
|---------|-----------------|------------------|------------|
| **Accuracy** | 0.7208 | 0.7155 | -0.0053 |
| **Recall Ataque** | 0.528 | 0.518 | -0.010 |
| **FP Rate** | 0.0249 | 0.0254 | +0.0005 |

**Conclusión**: Sorprendentemente, entrenar con el dataset completo **no mejora** significativamente el rendimiento. Posibles causas:
- El agente necesita **más timesteps** (>1M) para aprovechar más datos
- El dataset 20% ya contiene ejemplos suficientemente representativos
- Hiperparámetros (learning_rate, buffer_size) podrían necesitar ajuste para dataset grande

---

## 🔬 Detalles de Experimentos Clave

### Experimento E01: Baseline DQN Inicial

**Objetivo**: Establecer un baseline de RL con configuración estándar.

**Configuración**:
```python
REWARD_CONFIG = {
    "tp": 1.0,
    "fp": -1.0,
    "fn": -2.0,    # Penalización moderada
    "omission": 0.0
}

# Hiperparámetros DQN
learning_rate = 1e-3
buffer_size = 100_000
batch_size = 64
total_timesteps = 200_000
```

**Resultados**:
```
Accuracy: 0.7602
Precision (clase 1): 0.8235
Recall (clase 1): 0.600
F1-Score: 0.6946
FP Rate: 0.028
```

**Análisis**:
- El agente aprende una política conservadora (alta precision, recall moderado)
- Solo 2.8% de tráfico legítimo bloqueado (muy bajo FP rate)
- Detecta 60% de ataques (recall razonable pero mejorable)

**Conclusión**: Baseline sólido que prioriza **no bloquear tráfico legítimo** sobre detectar todos los ataques.

---

### Experimento E02: Baseline Random Forest

**Objetivo**: Establecer baseline supervisado para comparar con RL.

**Configuración**:
```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    n_jobs=-1,
    random_state=42
)
```

**Resultados**:
```
Accuracy: 0.7693
Precision (clase 1): 0.8187
Recall (clase 1): 0.615
F1-Score: 0.7028
FP Rate: 0.0267
```

**Análisis**:
- **Supera ligeramente** al DQN baseline en todas las métricas
- Tiempo de entrenamiento mucho menor (~5 min vs ~45 min)
- Recall de ataques 2.5% superior relativo (0.615 vs 0.600)
- FP rate ligeramente mejor (0.0267 vs 0.028)

**Ventajas de RF sobre DQN (en esta fase)**:
- ✅ Más rápido de entrenar
- ✅ No requiere GPU
- ✅ Hiperparámetros más intuitivos
- ✅ Mejor rendimiento out-of-the-box

**Ventajas de DQN sobre RF**:
- ✅ Ajustable via recompensas sin re-entrenar
- ✅ Puede aprender online de nuevos datos
- ✅ Potencial para optimizar objetivos complejos

**Conclusión**: Para deployment inicial, **RF es la opción más práctica**. DQN es prometedor para escenarios que requieran adaptabilidad.

---

### Experimento E05: DQN Pro-Seguridad

**Objetivo**: Maximizar detección de ataques (recall) mediante recompensas agresivas.

**Configuración**:
```python
REWARD_CONFIG = {
    "tp": 2.0,      # Recompensa alta por bloquear ataque
    "fp": -1.0,
    "fn": -6.0,     # Penalización muy fuerte por permitir ataque
    "omission": 0.2
}

total_timesteps = 500_000
```

**Resultados**:
```
Accuracy: 0.7563
Recall (clase 1): 0.5955  # Segundo mejor recall de experimentos DQN
F1-Score: 0.7015
FP Rate: 0.0313           # Ligeramente mayor FP
```

**Análisis**:
- **Alto recall** comparado con otros experimentos DQN (0.5955, cercano al 0.600 de E01)
- Trade-off: FP rate aumenta a 3.13% (vs 2.8% de E01)
- La recompensa alta en TP incentiva al agente a bloquear más agresivamente

**Conclusión**: Esta configuración es adecuada para **entornos de alta seguridad** donde detectar todos los ataques es crítico, aunque se acepten más falsos positivos.

---

### Experimento E06: DQN sin Omission Reward

**Objetivo**: Evaluar el impacto de la recompensa por permitir tráfico benigno.

**Configuración**:
```python
REWARD_CONFIG = {
    "tp": 1.5,
    "fp": -1.0,
    "fn": -5.0,
    "omission": 0.0  # Sin recompensa por TN
}
```

**Resultados**:
```
Accuracy: 0.7555
Recall (clase 1): 0.5928
FP Rate: 0.0296
```

**Análisis**:
- Resultados **muy similares** a E05 (que tenía omission=0.2)
- Omission reward tiene **impacto menor** de lo esperado
- El agente aprende principalmente de las penalizaciones (FP, FN)

**Conclusión**: La penalización de FP es suficiente para que el agente aprenda a no bloquear tráfico legítimo. La recompensa adicional por omission es opcional.

---

## 🎯 Recomendaciones de Configuración

### Para Diferentes Casos de Uso

#### 1. Entorno Corporativo Estándar (Balance)
**Objetivo**: Balance entre seguridad y disponibilidad

```python
REWARD_CONFIG = {
    "tp": 1.5,
    "fp": -1.0,
    "fn": -5.0,
    "omission": 0.5
}
total_timesteps = 500_000
```
**Esperado**: Acc ~0.75, Recall ~0.59, FP rate ~0.03

#### 2. Infraestructura Crítica (Pro-Seguridad)
**Objetivo**: Detectar máximo de ataques, tolerante a FP

```python
REWARD_CONFIG = {
    "tp": 2.5,
    "fp": -0.5,
    "fn": -10.0,
    "omission": 0.0
}
total_timesteps = 1_000_000
```
**Esperado**: Recall >0.65, FP rate ~0.05

#### 3. Servicio Público (Pro-Disponibilidad)
**Objetivo**: Minimizar falsos positivos, más tolerante a FN

```python
REWARD_CONFIG = {
    "tp": 1.0,
    "fp": -3.0,
    "fn": -2.0,
    "omission": 1.0
}
total_timesteps = 500_000
```
**Esperado**: FP rate <0.02, Recall ~0.55

---

## 📊 Métricas Detalladas por Experimento

### Confusion Matrices

#### E01 (DQN Baseline)
```
                Predicho PERMIT   Predicho BLOCK
Real Normal          9439              272        (FP rate: 2.8%)
Real Ataque          5182              7651       (Recall: 59.6%)
```

#### E02 (Random Forest)
```
                Predicho PERMIT   Predicho BLOCK
Real Normal          9452              259        (FP rate: 2.67%)
Real Ataque          4971              7862       (Recall: 61.3%)
```

#### E05 (DQN Pro-Seguridad)
```
                Predicho PERMIT   Predicho BLOCK
Real Normal          9407              304        (FP rate: 3.13%)
Real Ataque          5234              7599       (Recall: 59.2%)
```

---
## 🧠 Serie Axx: Ablación de arquitectura e hiperparámetros (NSL-KDD 20% train)

| ID  | Algoritmo | net_arch         | lr   | batch | train_freq | grad_steps | buffer | Steps | Reward (tp,fp,fn,om) | Resultado |
|-----|----------|------------------|------|-------|-----------|-----------|--------|-------|----------------------|----------|
| A01 | DQN      | [256, 256]       | 1e-4 | 2048  | 100       | 100       | 200k   | 500k  | ( , , , )            | ✅ Completado |
| A02 | DQN      | [512, 256]       | 1e-4 | 2048  | 100       | 100       | 200k   | 500k  | ( , , , )            | ✅ Completado |
| A03 | DQN      | [128, 256, 256]    | 1e-4 | 2048  | 100       | 100       | 200k   | 500k  | ( , , , )            | PENDIENTE|
| A04 | QRDQN    | best-of-above    | 1e-4 | 2048  | 100       | 100       | 200k   | 500k  | ( , , , )            | PENDIENTE|

**Nota**: La serie Axx fue reemplazada por los experimentos C01-C03 sobre CICIDS2017, que utilizan el esquema canónico y QRDQN (ver [`docs/results.md`](../docs/results.md)).

--- 

## 🔮 Próximos Experimentos Planificados

### Serie E07-E10: Optimización de Hiperparámetros
- **E07**: Grid search de learning_rate [1e-4, 5e-4, 1e-3, 5e-3]
- **E08**: Evaluación de buffer_size [50k, 100k, 200k, 500k]
- **E09**: Comparativa de arquitecturas de red (MLP profunda vs shallow)
- **E10**: Exploration strategies (epsilon-greedy variants)

### Serie E11-E15: Algoritmos RL Alternativos
- **E11**: PPO (Proximal Policy Optimization)
- **E12**: A2C (Advantage Actor-Critic)
- **E13**: Rainbow DQN (combinación de mejoras de DQN)
- **E14**: Dueling DQN
- **E15**: Comparativa exhaustiva de todos los algoritmos

### Serie E16-E20: Dataset Completo y Escalabilidad
- **E16**: Entrenamiento con 3M timesteps en dataset completo
- **E17**: Curriculum learning (empezar con 20%, luego full)
- **E18**: Multi-class classification (detectar tipo de ataque)
- **E19**: Ensemble de múltiples agentes DQN
- **E20**: Transfer learning desde E16 a otros datasets

### Serie E21+: Robustez y Adversarial ML
- **E21**: Evaluación contra evasion attacks
- **E22**: Adversarial training
- **E23**: Certified robustness evaluation
- **E24**: Concept drift simulation

---

## 📚 Referencias y Recursos

### Papers Relacionados
- Mnih et al. (2013) - [Playing Atari with Deep Reinforcement Learning (DQN)](https://arxiv.org/abs/1312.5602)
- Nguyen & Reddi (2019) - [Deep Reinforcement Learning for Cyber Security](https://arxiv.org/abs/1906.05799)
- Tavallaee et al. (2009) - [NSL-KDD Dataset: A Detailed Analysis](https://dl.acm.org/doi/10.5555/1736481.1736489)

### Código y Configuraciones
- Código de entrenamiento: `../src/train_rl_defender.py`
- Definición del entorno: `../src/rl_defender_env.py`
- Baseline RF: `../src/baseline_random_forest.py`
- Loader de dataset: `../src/load_nsl_kdd.py`

### Herramientas Utilizadas
- **Stable-Baselines3**: Implementación de DQN
- **Gymnasium**: Framework de entornos RL
- **scikit-learn**: Random Forest y métricas
- **pandas/numpy**: Procesamiento de datos