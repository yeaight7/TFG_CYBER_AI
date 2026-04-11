# TFG – Agente de Ciberseguridad con Aprendizaje por Refuerzo

Este repositorio contiene un Trabajo Fin de Grado orientado al diseño de un **agente defensor** basado en **Aprendizaje por Refuerzo (Reinforcement Learning, RL)** para tareas de detección y bloqueo de tráfico malicioso.

| Item | Detail |
|------|--------|
| **Dataset principal** | CICIDS2017 (~2.8 M flows, tráfico moderno con features extraíbles de PCAP) |
| **Algoritmo** | QRDQN (Quantile Regression DQN) — distributional RL via `sb3-contrib` |
| **Esquema canónico** | 76 flow features + 76 missingness mask → **152-dim observation** |
| **Mejor modelo** | Accuracy 0.9986, Recall ataque 0.9995, F1 0.9988 ([resultados completos](docs/results.md)) |
| **Validación** | Check A/B/C + leave-one-exact-CSV-out sobre los 8 CSVs reales de CICIDS2017 |

---

## Ejecución rápida

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Smoke test (~2-5 min, 50k rows, 5k timesteps)
python src/train_rl_defender.py --smoke

# 3. Entrenamiento completo (~30-60 min, 250k rows, 100k timesteps)
python src/train_rl_defender.py --preset full

# 4. Entrenamiento con parámetros custom
python src/train_rl_defender.py --timesteps 200000 --max-rows 500000

# 5. Split por día (train Mon-Wed, test Thu-Fri)
python src/train_rl_defender.py --split-mode day

# 6. Sin esquema canónico (features raw)
python src/train_rl_defender.py --smoke --no-canonical

# 7. Optimización de hiperparámetros con Optuna
python src/tune_hparams.py --n-trials 20 --timesteps 10000

# 8. Validation checks (A=direct eval, B=shuffled labels, C=CSV split)
python src/validate_checks.py --model models/<MODEL>.zip --checks A B C

# 9. Leave-one-exact-CSV-out sobre los CSVs reales de CICIDS2017
python src/validate_leave_one_csv_out.py --timesteps 30000

# 10. Smoke/dev run de leave-one-exact-CSV-out
python src/validate_leave_one_csv_out.py --timesteps 5000 --max-rows-per-csv 10000

# 11. Ver resultados con TensorBoard
tensorboard --logdir runs/cicids2017/
```

Todos los resultados se guardan en `runs/<category>/<RUN_ID>/` con `config.json` y `metrics.json`.
Los modelos se guardan en `models/<RUN_ID>.zip`.

---

## 📁 Estructura del Proyecto

```text
TFG_CYBER_AI/
├── src/
│   ├── canonical_schema.py        # 76 features canónicas + mappings + missingness mask
│   ├── load_cicids2017.py         # Adapter CICIDS2017 (dataset principal)
│   ├── load_nsl_kdd.py            # Adapter NSL-KDD (benchmark histórico)
│   ├── rl_defender_env.py         # Entorno Gymnasium custom (152-dim obs, Discrete(2))
│   ├── train_rl_defender.py       # Entrenamiento QRDQN con --smoke / --preset full
│   ├── validate_checks.py        # Checks A/B/C de validación
│   ├── validate_leave_one_csv_out.py # Validación leave-one-exact-CSV-out
│   ├── tune_hparams.py           # Optimización de hiperparámetros con Optuna
│   ├── scaling_utils.py          # Utilidades de escalado de features
│   └── baseline_random_forest.py # Baseline supervisado con Random Forest
│
├── scripts/
│   ├── predict_real_traffic.py    # Inferencia Phase 2 (v1, legacy)
│   └── predict_real_traffic_v2.py # Inferencia Phase 2 (v2, robusta con z-clipping)
│
├── lab/
│   └── docker/                    # Docker Compose para lab privado (nginx + generador)
│
├── pcaps/                         # PCAPs capturados y CSVs de flows extraídos
│
├── datasets/
│   └── CICIDS2017/                # Dataset CICIDS2017 (8 CSVs, ~2.8M flows)
│
├── models/                        # Modelos entrenados (.zip, .joblib)
├── runs/                          # Resultados por experimento
│   ├── cicids2017/                #   Runs de entrenamiento QRDQN (C01, C02, C03)
│   ├── validation/                #   Runs de validation checks (A, B, C)
│   ├── phase2/                    #   Runs de inferencia Phase 2 (lab traffic)
│   ├── nslkdd/                    #   Runs Phase 1 (NSL-KDD benchmark)
│   └── optuna/                    #   Estudios de hiperparámetros
├── experiments/                   # Documentación de experimentos
├── docs/                          # Documentación adicional
│   ├── results.md                 #   Métricas consolidadas (extraídas de JSON)
│   ├── phase2_plan.md             #   Plan paso a paso para Phase 2
│   ├── AGENT_CONTEXT.md           #   Contexto Phase 2 para coding agents
│   └── gcp_lab.md                 #   Instrucciones de lab privado (GCP)
├── report/                        # Memoria del TFG (LaTeX)
├── requirements.txt               # Dependencias Python
└── README.md
```

---

## 🚀 Instalación y Configuración

### Requisitos Previos

- **Python 3.10+** (recomendado 3.10 o 3.11)
- **pip** (gestor de paquetes de Python)
- Dataset CICIDS2017 en `datasets/CICIDS2017/` (8 archivos CSV)

### Instalación

```bash
git clone https://github.com/yeaight7/TFG_CYBER_AI.git
cd TFG_CYBER_AI
python3 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🏗️ Arquitectura del Sistema

### Esquema Canónico de Features

El proyecto usa un **esquema canónico fijo** de 76 features flow-based (definidas en `src/canonical_schema.py`) que:

1. Existen en CICIDS2017 (dataset principal)
2. Son extraíbles de tráfico real/PCAP con CICFlowMeter/Zeek
3. No causan data leakage (sin IPs, timestamps, Flow IDs)

Cada observación del agente es un vector de **152 dimensiones**:

```
obs = [x_1, x_2, ..., x_76, m_1, m_2, ..., m_76]
```

Donde `x_i` es el valor de la feature (imputado con 0 si falta) y `m_i` indica si estaba presente (1) o ausente (0).

### Entorno RL (Gymnasium)

- **Espacio de observación**: `Box(152,)` — vector de features + máscara de missingness
- **Espacio de acciones**: `Discrete(2)` — 0 = PERMIT, 1 = BLOCK
- **Sistema de recompensas** (configurable):
  - TP (bloquear ataque): +1.5
  - FP (bloquear benigno): −1.0
  - FN (permitir ataque): −5.0
  - TN (permitir benigno): 0.0

### Agente RL (QRDQN)

- **Algoritmo**: Quantile Regression DQN (`sb3-contrib`)
- **Red**: MLP [512, 256]
- **Learning rate**: 1 × 10⁻⁴
- **Batch size**: 2 048 (full) / 256 (smoke)

### Validation Checks

| Check | Descripción |
|-------|-------------|
| **A** | Evaluación directa `model.predict(X_test[i])` vs `y_test[i]`, sin depender del entorno |
| **B** | Entrena con labels barajados → confirma que accuracy cae a nivel aleatorio (sin leakage) |
| **C** | Split por CSV: entrena en Mon–Wed, testea en Thu–Fri (generalización real) |
| **Leave-One-CSV-Out** | Entrena 8 folds dejando fuera 1 CSV exacto en cada run y agregando métricas por fold |

Ver resultados reales en [`docs/results.md`](docs/results.md).

---

## 📊 Resultados Actuales

Los resultados completos con métricas extraídas de los JSON de cada run están en [`docs/results.md`](docs/results.md).

### Resumen (CICIDS2017 — QRDQN)

| Run | Rows | Timesteps | Accuracy | Recall atk | F1 atk |
|-----|------|-----------|----------|------------|--------|
| C01 smoke | 50k | 5k | 0.9697 | 0.9996 | 0.9692 |
| C01 full | 250k | 100k | 0.9962 | 0.9998 | 0.9963 |
| C02 fast | 100k | 10k | 0.9766 | 0.9996 | 0.9812 |
| **C03 full** | **500k** | **100k** | **0.9986** | **0.9995** | **0.9988** |

### Validation Check Highlights

| Check | Key Result |
|-------|------------|
| A (direct eval) | Accuracy 0.9939 — TP=4772, FP=60, FN=1 |
| B (anti-leakage) | Shuffled acc 0.4773 vs baseline 0.5227 → ✅ no leakage |
| C (CSV-split) | Accuracy 0.8414 (30k timesteps, unseen days) |

---

## 🔬 Experimentación

### Experiments Phase 1 (NSL-KDD)

Consulta [`experiments/nslkdd_experiments.md`](experiments/nslkdd_experiments.md) para los resultados de Phase 1 con DQN y Random Forest sobre NSL-KDD.

### Ajustar Sistema de Recompensas

```python
REWARD_CONFIG = {
    "tp": 1.5,      # Recompensa por bloquear ataque (True Positive)
    "fp": -1.0,     # Penalización por bloquear tráfico legítimo (False Positive)
    "fn": -5.0,     # Penalización fuerte por permitir ataque (False Negative)
    "omission": 0.0  # Recompensa por permitir tráfico legítimo (True Negative)
}
```

### Registro de Runs

Cada run produce:
```
runs/<category>/<RUN_ID>/
├── config.json      # configuración completa
├── metrics.json     # métricas finales
└── ...              # TensorBoard logs, etc.
```

---

## 🔮 Trabajo Futuro

### Phase 2: Entorno Simulado con Tráfico Real

Plan detallado en [`docs/phase2_plan.md`](docs/phase2_plan.md). Instrucciones de lab en [`docs/gcp_lab.md`](docs/gcp_lab.md).

1. Desplegar lab privado (2 VMs: Kali attacker + Ubuntu defender)
2. Generar tráfico labelled (benigno + ataques)
3. Capturar PCAPs y extraer flow features con CICFlowMeter
4. Mapear al esquema canónico (152 dims)
5. Inference loop con el modelo QRDQN entrenado
6. Evaluar contra ground-truth

### Phase 3+

- Adversario RL (attacker agent)
- Multi-agente y defensa distribuida
- Despliegue productivo (Docker, CI/CD)

---

## 🤝 Contribuciones

Este es un proyecto académico (TFG), pero se aceptan sugerencias y mejoras:

1. Fork del repositorio
2. Crea una rama: `git checkout -b feature/nueva-mejora`
3. Implementa cambios siguiendo las convenciones en [`.github/copilot-instructions.md`](.github/copilot-instructions.md)
4. Abre PR con descripción detallada

---

## 📚 Referencias

- [CICIDS2017 Dataset](https://www.unb.ca/cic/datasets/ids-2017.html)
- [NSL-KDD Dataset](https://www.unb.ca/cic/datasets/nsl.html)
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [sb3-contrib (QRDQN)](https://sb3-contrib.readthedocs.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Quantile Regression DQN Paper](https://arxiv.org/abs/1710.10044)
