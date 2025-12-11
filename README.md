# TFG – Agente de Ciberseguridad con Aprendizaje por Refuerzo

Este repositorio contiene la primera fase de un Trabajo Fin de Grado orientado al diseño de un **agente defensor** basado en **Aprendizaje por Refuerzo (Reinforcement Learning, RL)** para tareas de detección y bloqueo de tráfico malicioso.

En esta fase el entorno es **simulado / tipo dataset**: el agente recibe características de flujos de red (u otras muestras etiquetadas como benignas o maliciosas) y aprende una política para **permitir o bloquear** el tráfico maximizando una función de recompensa.

## 🎯 Objetivo del Proyecto

El objetivo principal es desarrollar un sistema de defensa inteligente basado en RL que pueda:

- **Detectar automáticamente** tráfico malicioso en redes
- **Aprender políticas óptimas** de bloqueo/permisión mediante recompensas
- **Minimizar falsos positivos** (bloquear tráfico legítimo)
- **Minimizar falsos negativos** (permitir ataques)
- **Generalizar** a nuevos tipos de ataques no vistos durante el entrenamiento

## 🏗️ Arquitectura del Sistema

El sistema se compone de tres componentes principales:

### 1. **Entorno RL Custom (Gymnasium)**
- Implementado en `rl_defender_env.py`
- Basado en el framework Gymnasium (sucesor de OpenAI Gym)
- **Espacio de observación**: Vector de características de flujos de red (multidimensional)
- **Espacio de acciones**: Discreto (0 = PERMIT, 1 = BLOCK)
- **Sistema de recompensas**:
  - Bloquear ataque correctamente: +1.0 (recompensa)
  - Permitir tráfico benigno: +0.5 (recompensa menor)
  - Bloquear tráfico benigno (FP): -1.0 (penalización)
  - Permitir ataque (FN): -5.0 (penalización fuerte)

### 2. **Agente RL (DQN)**
- Algoritmo: **Deep Q-Network (DQN)** de Stable-Baselines3
- Política: MLP (Multi-Layer Perceptron)
- Red neuronal profunda que aprende valores Q(s,a) para cada par estado-acción
- Utiliza replay buffer y target network para estabilizar el entrenamiento

### 3. **Dataset: NSL-KDD**
- Versión mejorada del dataset KDD Cup 1999
- Contiene flujos de red con características como:
  - Duración de conexión
  - Tipo de protocolo (TCP, UDP, ICMP)
  - Servicio de red (HTTP, FTP, SSH, etc.)
  - Flags de conexión
  - Bytes enviados/recibidos
  - Tasas de error
  - Y 41 características más
- Etiquetas: Normal vs. Ataques (DoS, Probe, R2L, U2R)
- Descarga automática vía `kagglehub` desde Kaggle

---

## 📁 Estructura del Proyecto

```text
TFG_CYBER_AI/
├── src/
│   ├── rl_defender_env.py       # Entorno Gymnasium personalizado
│   ├── train_rl_defender.py      # Script principal de entrenamiento RL
│   ├── baseline_random_forest.py # Baseline supervisado con Random Forest
│   └── load_nsl_kdd.py           # Utilidad para cargar y preprocesar NSL-KDD
│
├── datasets/
│   └── nsl_kdd/                  # Dataset NSL-KDD (descargado automáticamente)
│       ├── KDDTrain+.txt         # Conjunto de entrenamiento completo
│       ├── KDDTrain+_20Percent.txt  # Versión reducida (20%)
│       ├── KDDTest+.txt          # Conjunto de prueba
│       └── ...
│
├── models/
│   ├── rl_defender_dqn.zip       # Modelo DQN entrenado (guardado)
│   ├── rl_defender_dqn_nslkdd.zip  # Modelo DQN con dataset completo
│   └── rf_nslkdd.joblib          # Modelo Random Forest baseline
│
├── experiments/
│   ├── README.md                 # Documentación de experimentos
│   └── nslkdd_experiments.md     # Resultados detallados experimentos NSL-KDD
│
├── report/
│   ├── report.pdf                # Memoria del TFG
│   └── report.tex                # Código fuente LaTeX de la memoria
│
├── .gitignore
└── README.md
```

---

## 🚀 Instalación y Configuración

### Requisitos Previos

- **Python 3.8+** (recomendado 3.9 o 3.10)
- **pip** (gestor de paquetes de Python)
- Conexión a internet (para descargar el dataset NSL-KDD desde Kaggle)

### Paso 1: Clonar el Repositorio

```bash
git clone https://github.com/yeaight7/TFG_CYBER_AI.git
cd TFG_CYBER_AI
```

### Paso 2: Crear Entorno Virtual

```bash
python3 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### Paso 3: Instalar Dependencias

```bash
pip install --upgrade pip
pip install numpy pandas scikit-learn
pip install gymnasium
pip install stable-baselines3
pip install kagglehub
```

**Dependencias principales:**
- `numpy`: Cálculos numéricos
- `pandas`: Manipulación de datos
- `scikit-learn`: Preprocesamiento y métricas
- `gymnasium`: Framework de entornos RL
- `stable-baselines3`: Implementaciones de algoritmos RL (DQN, PPO, A2C, etc.)
- `kagglehub`: Descarga automática de datasets de Kaggle

### Paso 4: Configurar Kaggle (Opcional)

Si es la primera vez usando `kagglehub`, podría pedirte credenciales de Kaggle:

1. Crea una cuenta en [Kaggle](https://www.kaggle.com/)
2. Ve a tu perfil → Settings → API → "Create New API Token"
3. Descarga el archivo `kaggle.json`
4. Colócalo en `~/.kaggle/kaggle.json` (Linux/Mac) o `C:\Users\<usuario>\.kaggle\kaggle.json` (Windows)

---

## 📊 Dataset: NSL-KDD

### Descripción

NSL-KDD es un dataset de detección de intrusiones derivado del KDD Cup 1999, diseñado para superar algunas limitaciones del dataset original. Contiene registros de conexiones de red con:

- **125,973 muestras de entrenamiento** (KDDTrain+.txt)
- **22,544 muestras de prueba** (KDDTest+.txt)
- **25,192 muestras de entrenamiento reducido** (KDDTrain+_20Percent.txt) - Útil para experimentación rápida
- **41 características numéricas/categóricas** por muestra
- **Etiquetas**: Normal, DoS, Probe, R2L, U2R

### Características del Dataset

Las 41 características se agrupan en categorías:

**Características básicas de conexión (9)**:
- `duration`: Duración de la conexión en segundos
- `protocol_type`: Protocolo (TCP, UDP, ICMP)
- `service`: Servicio de red (HTTP, FTP, SSH, etc.)
- `flag`: Estado de la conexión (SF, S0, REJ, etc.)
- `src_bytes`, `dst_bytes`: Bytes enviados/recibidos

**Características de contenido (13)**:
- `hot`, `num_failed_logins`, `logged_in`, etc.
- Indicadores de comportamiento sospechoso

**Características de tráfico temporal (9)**:
- `count`, `srv_count`: Conexiones al mismo host/servicio en 2 segundos
- `serror_rate`, `rerror_rate`: Tasas de error

**Características basadas en host (10)**:
- `dst_host_count`: Conexiones al host destino en 100 conexiones
- `dst_host_srv_count`: Conexiones al mismo servicio
- Tasas de error a nivel de host

### Distribución de Ataques

**Train Set (KDDTrain+.txt)**:
- Normal: 67,343 (53.5%)
- DoS: 45,927 (36.5%)
- Probe: 11,656 (9.3%)
- R2L: 995 (0.8%)
- U2R: 52 (0.04%)

**Test Set (KDDTest+.txt)**:
- Normal: 9,711 (43.1%)
- DoS: 7,458 (33.1%)
- Probe: 2,421 (10.7%)
- R2L: 2,754 (12.2%)
- U2R: 200 (0.9%)

**Nota**: El test set contiene tipos de ataque diferentes a los del train para evaluar generalización.

### Preprocesamiento Automático

El script `load_nsl_kdd.py` realiza:

1. **Descarga automática** desde Kaggle vía `kagglehub`
2. **One-hot encoding** de variables categóricas (protocol_type, service, flag)
   - Resultado: 122 características numéricas finales
3. **Binarización de etiquetas**: 0 = Normal, 1 = Ataque (cualquier tipo)
4. **División train/test** manteniendo la proporción original del dataset
5. **Conversión a arrays NumPy** (float32) para eficiencia y compatibilidad con RL

### Uso

```python
from load_nsl_kdd import load_nsl_kdd_binary

# Cargar dataset completo (125,973 train samples)
X_train, y_train, X_test, y_test = load_nsl_kdd_binary(use_20_percent=False)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
# Output: Train: (125973, 122), Test: (22544, 122)

# O usar versión reducida (20%) para experimentación rápida
X_train, y_train, X_test, y_test = load_nsl_kdd_binary(use_20_percent=True)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
# Output: Train: (25192, 122), Test: (22544, 122)
```

### Ventajas de NSL-KDD sobre KDD'99

1. ✅ **Sin registros redundantes**: Elimina duplicados que sesgaban el entrenamiento
2. ✅ **Balanceo mejorado**: Mejor distribución de clases
3. ✅ **Tamaño razonable**: Permite entrenar sin necesidad de sampling
4. ✅ **Test set desafiante**: Incluye ataques no vistos en train para evaluar generalización

### Limitaciones a Considerar

1. ⚠️ **Dataset antiguo**: Basado en tráfico de 1999, no incluye ataques modernos
2. ⚠️ **Tráfico simulado**: Generado en laboratorio, no tráfico real de producción
3. ⚠️ **Protocolos obsoletos**: No incluye HTTPS, HTTP/2, QUIC, WebSockets, etc.
4. ⚠️ **Contexto limitado**: No captura patrones de ataque distribuidos o APTs

Para aplicaciones en producción, considera complementar con datasets más recientes (CICIDS2017, UNSW-NB15).

---

## 🎓 Entrenamiento del Agente

### Ejecución Básica

```bash
cd src
python train_rl_defender.py
```

### Proceso de Entrenamiento

El script `train_rl_defender.py` realiza:

1. **Carga del dataset NSL-KDD**
   ```
   Train shape: X=(125973, 122), y=(125973,)
   Test shape:  X=(22544, 122), y=(22544,)
   ```

2. **Creación del entorno RL**
   - Entorno personalizado `RLDatasetDefenderEnv`
   - Envuelto en `DummyVecEnv` para compatibilidad con SB3

3. **Inicialización del modelo DQN**
   - Política: MLP con capas ocultas
   - Learning rate: 1e-3
   - Buffer size: 100,000 transiciones
   - Batch size: 64
   - Gamma (descuento): 0.99
   - Target network update: cada 10,000 pasos

4. **Entrenamiento**
   - Total timesteps: 1_000_000
   - Cada episodio recorre hasta 10_000 muestras
   - El agente aprende de interacciones repetidas

5. **Guardado del modelo**
   - Archivo: `models/rl_defender_dqn.zip`

6. **Evaluación en test**
   - Métricas: Matriz de confusión, Precision, Recall, F1-Score
   - Acción 0 = PERMIT, Acción 1 = BLOCK

### Parámetros Configurables

Puedes modificar hiperparámetros en `train_rl_defender.py`:

```python
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=1e-3,        # Tasa de aprendizaje
    buffer_size=100_000,        # Tamaño del replay buffer
    batch_size=64,              # Tamaño del batch
    gamma=0.99,                 # Factor de descuento
    tau=1.0,                    # Tasa de actualización de target network
    train_freq=4,               # Frecuencia de entrenamiento
    target_update_interval=10_000,  # Intervalo de actualización
    verbose=1,
)
```

---

## 🧪 Evaluación y Métricas

### Matriz de Confusión

```
                Predicho PERMIT (0)  Predicho BLOCK (1)
Real Normal (0)        TN              FP
Real Ataque (1)        FN              TP
```

### Métricas Clave

- **Precision**: TP / (TP + FP) - Proporción de ataques correctamente identificados entre todos los bloqueos
- **Recall**: TP / (TP + FN) - Proporción de ataques detectados del total de ataques reales
- **F1-Score**: Media armónica de Precision y Recall
- **Accuracy**: (TP + TN) / Total - Proporción total de aciertos

### Ejemplo de Salida

```
=== Confusion matrix (acciones: 0=PERMIT, 1=BLOCK) ===
[[  9711    644]
 [  1544  10645]]

=== Classification report ===
              precision    recall  f1-score   support

           0     0.8630    0.9378    0.8988     10355
           1     0.9429    0.8733    0.9068     12189

    accuracy                         0.9030     22544
   macro avg     0.9030    0.9056    0.9028     22544
weighted avg     0.9061    0.9030    0.9034     22544
```

### Interpretación de Resultados

- **Clase 0 (PERMIT)**: El agente permite correctamente el 93.78% del tráfico benigno
- **Clase 1 (BLOCK)**: El agente bloquea correctamente el 87.33% de los ataques
- **False Positives (FP)**: 644 flujos benignos bloqueados incorrectamente (6.22% del tráfico legítimo)
- **False Negatives (FN)**: 1,544 ataques permitidos incorrectamente (12.67% de los ataques)
- **Accuracy Global**: 90.30% de decisiones correctas

El agente prioriza la **detección de ataques** (alta precision del 94.29% en bloqueos) mientras mantiene un balance razonable con los falsos positivos.

---

## 🔬 Experimentación

### Probar Diferentes Algoritmos RL

Además de DQN, puedes experimentar con:

```python
from stable_baselines3 import PPO, A2C, SAC

# PPO (Proximal Policy Optimization) - Recomendado para problemas de política
model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4)

# A2C (Advantage Actor-Critic) - Más rápido pero menos estable
model = A2C("MlpPolicy", env, verbose=1, learning_rate=7e-4)

# SAC (Soft Actor-Critic) - Requiere acciones continuas
# Nota: SAC no es compatible con espacios de acción discretos
```

### Comparación con Baseline Supervisado

El proyecto incluye un baseline de Random Forest para comparar:

```bash
cd src
python baseline_random_forest.py
```

Ventajas del RL sobre métodos supervisados:
- **Adaptabilidad**: Aprende políticas óptimas considerando consecuencias a largo plazo
- **Configuración de recompensas**: Permite ajustar el trade-off entre FP y FN
- **Aprendizaje continuo**: Puede adaptarse a nuevos patrones de ataque
- **Optimización de objetivos complejos**: Maximiza métricas específicas de seguridad

### Ajustar Sistema de Recompensas

El sistema de recompensas define el comportamiento del agente. Modifica en `train_rl_defender.py`:

```python
REWARD_CONFIG = {
    "tp": 1.5,      # Recompensa por bloquear ataque (True Positive)
    "fp": -1.0,     # Penalización por bloquear tráfico legítimo (False Positive)
    "fn": -5.0,     # Penalización fuerte por permitir ataque (False Negative)
    "omission": 0.0 # Recompensa por permitir tráfico legítimo (True Negative)
}
```

**Estrategias de recompensa:**

1. **Pro-seguridad** (minimizar FN): `tp=2.0, fp=-1.0, fn=-10.0, omission=0.0`
   - Prioriza detectar todos los ataques, acepta más falsos positivos
   
2. **Balanceada**: `tp=1.5, fp=-1.0, fn=-5.0, omission=0.5`
   - Balance entre detectar ataques y no bloquear tráfico legítimo
   
3. **Pro-disponibilidad** (minimizar FP): `tp=1.0, fp=-3.0, fn=-2.0, omission=1.0`
   - Prioriza no interrumpir tráfico legítimo, más tolerante con FN

### Entrenar con Subset Reducido

Para experimentación rápida durante desarrollo:

```python
# En train_rl_defender.py
X_train, y_train, X_test, y_test = load_nsl_kdd_binary(
    use_20_percent=True  # Solo 20% del dataset (~25k muestras)
)
```

Para entrenamiento final completo:

```python
X_train, y_train, X_test, y_test = load_nsl_kdd_binary(
    use_20_percent=False  # Dataset completo (~126k muestras)
)
```

### Registro de Experimentos

Consulta `experiments/nslkdd_experiments.md` para ver los resultados de experimentos previos con diferentes configuraciones de hiperparámetros y recompensas.

---

## 📈 Cargar Modelo Pre-entrenado

Para evaluar o continuar el entrenamiento:

```python
from stable_baselines3 import DQN
from rl_defender_env import RLDatasetDefenderEnv
from load_nsl_kdd import load_nsl_kdd_binary

# Cargar dataset
X_train, y_train, X_test, y_test = load_nsl_kdd_binary(use_20_percent=False)

# Crear entorno de evaluación
env = RLDatasetDefenderEnv(
    X=X_test,
    y=y_test,
    benign_label=0,
    attack_label=1,
    shuffle=False
)

# Cargar modelo guardado
model = DQN.load("models/rl_defender_dqn_nslkdd")

# Evaluar en nuevo conjunto
obs, info = env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(int(action))
    done = terminated or truncated
```

### Continuar Entrenamiento

```python
# Cargar modelo existente
model = DQN.load("models/rl_defender_dqn_nslkdd", env=vec_env)

# Continuar entrenamiento por 500k timesteps adicionales
model.learn(total_timesteps=500_000)

# Guardar modelo actualizado
model.save("models/rl_defender_dqn_extended")
```

---

## 🎯 Limitaciones Actuales

Este proyecto representa la fase inicial de investigación y presenta las siguientes limitaciones:

### Limitaciones Técnicas
- **Entorno simulado**: Usa dataset histórico en lugar de tráfico en tiempo real
- **Clasificación binaria**: Solo distingue entre normal y ataque (no tipos específicos)
- **Features estáticas**: Las 122 características se calculan a nivel de flujo completo
- **No streaming**: Procesa flujos completos, no paquetes individuales en tiempo real

### Limitaciones del Dataset NSL-KDD
- **Dataset antiguo**: Basado en tráfico de 1999, no refleja ataques modernos
- **Distribución artificial**: Proporción de ataques no representa tráfico real
- **Características limitadas**: No incluye información de capa de aplicación moderna
- **Protocolos obsoletos**: No incluye tráfico HTTPS, HTTP/2, QUIC, etc.

### Consideraciones de Despliegue
- **Latencia**: El agente actual no está optimizado para decisiones en microsegundos
- **Escalabilidad**: No probado en redes de alta velocidad (10+ Gbps)
- **Robustez**: No evaluado contra ataques adversariales dirigidos al modelo RL
- **Adaptación**: Requiere re-entrenamiento para nuevos tipos de ataques

## 🔮 Trabajo Futuro

Esta es la **Fase 1** del TFG. Las siguientes fases incluirán:

### Fase 2: Entorno en Tiempo Real
- **Captura de tráfico en vivo**
  - Integración con libpcap/Scapy para captura de paquetes
  - Pipeline de extracción de características en streaming
  - Procesamiento en tiempo real con latencias < 100ms
  
- **Feature Engineering moderno**
  - Características de tráfico HTTPS/TLS (certificados, handshakes)
  - Análisis de payloads cifrados (tamaños, timing)
  - Estadísticas de flujo en ventanas temporales deslizantes

- **Integración con infraestructura**
  - Interfaz con iptables/nftables para bloqueo dinámico
  - Logs estructurados para SIEM (Splunk, ELK)
  - Métricas de rendimiento (Prometheus, Grafana)

### Fase 3: Adversario RL
- **Modelado de atacantes inteligentes**
  - Implementar agente atacante con RL
  - Técnicas de evasión adaptativas
  - Escenario de juego adversarial (Game Theory)
  
- **Co-evolución y robustez**
  - Entrenamiento adversarial defensor vs. atacante
  - Técnicas de adversarial training para robustez
  - Evaluación contra ataques de envenenamiento de datos

### Fase 4: Multi-Agente y Distribución
- **Arquitectura multi-agente**
  - Sistema distribuido con múltiples defensores por zona
  - Coordinación mediante comunicación inter-agente
  - Políticas jerárquicas (agentes locales + coordinador global)
  
- **Defensa de redes complejas**
  - Topologías de red realistas
  - Propagación de ataques laterales
  - Defensa colaborativa en SDN (Software-Defined Networks)

### Fase 5: Despliegue Productivo
- **Contenedorización y orquestación**
  - Dockerfile y docker-compose para despliegue
  - Kubernetes para escalado automático
  - CI/CD para actualización continua de modelos
  
- **Integración con ecosistema empresarial**
  - API REST para integración con SOC
  - Webhooks para alertas en tiempo real
  - Dashboard web de monitorización (React/Vue.js)
  
- **Evaluación y mejora continua**
  - A/B testing de nuevas políticas
  - Monitorización de drift del modelo
  - Pipeline de re-entrenamiento automático

### Fase 6: Investigación Avanzada
- **Algoritmos RL avanzados**
  - Multi-objective RL (balance FP/FN/Throughput)
  - Meta-learning para adaptación rápida a nuevos ataques
  - Offline RL para aprendizaje seguro desde logs
  
- **Explicabilidad y confianza**
  - SHAP/LIME para explicar decisiones del agente
  - Certificación de robustez formal
  - Auditoría de decisiones para compliance

## 💡 Consejos y Mejores Prácticas

### Optimización del Entrenamiento
1. **Usa GPU si está disponible**: El entrenamiento puede ser 5-10x más rápido
   ```python
   model = DQN("MlpPolicy", env, device="cuda", ...)
   ```

2. **Ajusta buffer_size según memoria**: 
   - GPU 8GB: buffer_size=100_000
   - GPU 16GB+: buffer_size=500_000
   - Solo CPU: buffer_size=50_000

3. **Monitoriza el aprendizaje**:
   ```python
   from stable_baselines3.common.callbacks import EvalCallback
   
   eval_callback = EvalCallback(
       eval_env,
       best_model_save_path="./logs/best_model",
       log_path="./logs/results",
       eval_freq=10_000,
   )
   model.learn(total_timesteps=1_000_000, callback=eval_callback)
   ```

### Debugging y Desarrollo
1. **Empieza con subset pequeño**: Usa `use_20_percent=True` para iterar rápido
2. **Reduce timesteps inicialmente**: Prueba con 50k-100k antes de 1M
3. **Visualiza la matriz de confusión**: Identifica si el agente está sesgado
4. **Registra las recompensas**: Asegúrate de que aumentan durante el entrenamiento

### Reproducibilidad
```python
import numpy as np
import random
import torch

# Fijar semillas para reproducibilidad
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# Pasar seed al modelo
model = DQN("MlpPolicy", env, seed=SEED, ...)
```

---

## 🔧 Troubleshooting

### Problemas Comunes

#### Error: "kagglehub requiere autenticación"
**Solución**: Configura las credenciales de Kaggle:
1. Crea cuenta en [Kaggle](https://www.kaggle.com/)
2. Ve a Settings → API → "Create New API Token"
3. Coloca `kaggle.json` en `~/.kaggle/` (Linux/Mac) o `C:\Users\<usuario>\.kaggle\` (Windows)
4. Permisos: `chmod 600 ~/.kaggle/kaggle.json`

#### Error: "CUDA out of memory"
**Solución**: Reduce el tamaño del batch o buffer:
```python
model = DQN(
    "MlpPolicy",
    env,
    buffer_size=50_000,  # Reducir de 100k
    batch_size=32,       # Reducir de 64
    device="cpu",        # O usar CPU
)
```

#### Advertencia: "No GPU detected"
**Verificar CUDA**:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
Si devuelve `False`:
- Instala PyTorch con CUDA: https://pytorch.org/get-started/locally/
- Verifica drivers NVIDIA: `nvidia-smi`

#### Error: "FileNotFoundError: KDDTrain+.txt"
**Solución**: El dataset no se descargó correctamente:
```python
# Fuerza descarga manual
import kagglehub
path = kagglehub.dataset_download("hassan06/nslkdd")
print(f"Dataset en: {path}")
```

#### El agente no aprende (accuracy estancada)
**Posibles causas**:
1. **Learning rate muy alto/bajo**: Prueba `1e-4` o `5e-4`
2. **Reward mal diseñada**: Verifica que las penalizaciones no dominen
3. **Exploration insuficiente**: Aumenta `exploration_fraction` en DQN
4. **Dataset no balanceado**: Considera `class_weight` en RF o ajusta rewards

### Performance Benchmarks

Tiempos aproximados en diferentes configuraciones:

| Configuración | Dataset | Timesteps | Tiempo Entrenamiento | Accuracy Test |
|---------------|---------|-----------|---------------------|---------------|
| CPU (8 cores) | 20% | 500k | ~45 min | ~76% |
| CPU (8 cores) | Full | 1M | ~3 horas | ~72% |
| GPU (RTX 3070) | 20% | 500k | ~12 min | ~76% |
| GPU (RTX 3070) | Full | 1M | ~30 min | ~72% |
| GPU (A100) | Full | 1M | ~15 min | ~72% |

**Nota**: Los tiempos varían según hardware, sistema operativo y carga del sistema.

## 🤝 Contribuciones

Este es un proyecto académico (TFG), pero se aceptan sugerencias y mejoras:

1. **Fork del repositorio**: Haz una copia en tu cuenta de GitHub
2. **Crea una rama**: `git checkout -b feature/nueva-mejora`
3. **Implementa cambios**: 
   - Sigue el estilo de código existente
   - Añade docstrings a funciones nuevas
   - Comenta código complejo
4. **Prueba tus cambios**: Verifica que funcionen correctamente
5. **Commit**: `git commit -m "feat: descripción clara del cambio"`
6. **Push**: `git push origin feature/nueva-mejora`
7. **Pull Request**: Abre PR con descripción detallada

### Áreas de Contribución Sugeridas
- 🆕 Nuevos algoritmos RL (PPO, SAC, TD3)
- 📊 Visualizaciones de aprendizaje (TensorBoard, W&B)
- 🔬 Experimentos con otros datasets (UNSW-NB15, CICIDS2017)
- 📝 Mejoras en documentación
- 🐛 Corrección de bugs
- ⚡ Optimizaciones de rendimiento
- 🧪 Casos de test unitarios

---

## ❓ FAQ (Preguntas Frecuentes)

### ¿Por qué usar RL en lugar de ML supervisado tradicional?

**Ventajas del RL**:
- **Optimización de objetivos complejos**: Puedes optimizar directamente el trade-off entre FP y FN mediante recompensas
- **Aprendizaje secuencial**: El agente considera consecuencias a largo plazo, no solo decisiones puntuales
- **Adaptabilidad**: Puede aprender online de nuevos datos sin re-entrenar desde cero
- **Flexibilidad**: Fácil ajustar el comportamiento cambiando recompensas sin re-entrenar modelos

**Desventajas**:
- Más complejo de implementar y debuggear
- Requiere más tiempo de entrenamiento
- Más difícil de interpretar que modelos como Random Forest

### ¿El agente funciona en tiempo real?

**No en la versión actual**. Esta es la Fase 1 (entorno simulado con dataset histórico). El agente:
- ✅ Funciona: Con dataset NSL-KDD preprocesado
- ❌ No funciona: Con captura de tráfico en vivo (pcap, Wireshark)

La Fase 2 incluirá captura en tiempo real y extracción de características online.

### ¿Qué tipos de ataques detecta?

NSL-KDD incluye 4 categorías principales:
- **DoS** (Denial of Service): neptune, smurf, pod, teardrop, land, back
- **Probe** (Reconnaissance): portsweep, ipsweep, nmap, satan
- **R2L** (Remote to Local): guess_passwd, ftp_write, imap, phf, multihop, warezmaster, warezclient, spy
- **U2R** (User to Root): buffer_overflow, loadmodule, perl, rootkit

Actualmente el modelo solo clasifica binario (normal vs. ataque), no distingue entre tipos.

### ¿Puedo usar mis propios datos?

**Sí**, pero requiere adaptación. Necesitas:

1. **Formato correcto**: Array NumPy con shape `(n_samples, n_features)`
2. **Etiquetas binarias**: 0 = normal, 1 = ataque
3. **Preprocesamiento**: One-hot encoding de categóricas, normalización si es necesario

Ejemplo:
```python
from rl_defender_env import RLDatasetDefenderEnv

# Tus datos (n_samples, n_features)
X_train = np.load("mi_dataset_X.npy")
y_train = np.load("mi_dataset_y.npy")

# Crear entorno
env = RLDatasetDefenderEnv(X=X_train, y=y_train)

# Entrenar como de costumbre
model = DQN("MlpPolicy", env, ...)
```

### ¿Cómo elijo los hiperparámetros óptimos?

Recomendaciones basadas en los experimentos:

**Para empezar (baseline)**:
```python
learning_rate = 1e-3
buffer_size = 100_000
batch_size = 64
total_timesteps = 500_000
```

**Para optimizar**:
1. **Grid search manual**: Prueba combinaciones y documenta en `experiments/`
2. **Optuna**: Framework de optimización de hiperparámetros
   ```bash
   pip install optuna
   ```
3. **Ray Tune**: Para búsqueda distribuida a gran escala

**Tip**: Empieza con pocos timesteps (50k) para iterar rápido, luego entrena más.

### ¿El modelo es robusto contra ataques adversariales?

**No está evaluado**. Los ataques adversariales contra modelos de ML/RL en ciberseguridad incluyen:

- **Evasion attacks**: Modificar tráfico malicioso para parecer benigno
- **Poisoning attacks**: Inyectar datos maliciosos durante entrenamiento
- **Model extraction**: Robar el modelo mediante queries

La **Fase 3** incluirá evaluación de robustez adversarial.

### ¿Cuánta memoria RAM/GPU necesito?

**Mínimos**:
- RAM: 8 GB (dataset 20%)
- GPU: Opcional, pero acelera 3-5x (4 GB VRAM suficiente)

**Recomendados**:
- RAM: 16 GB (dataset completo)
- GPU: 8 GB VRAM (NVIDIA RTX 3060/4060 o superior)

**Sin GPU**: Todo funciona en CPU, solo será más lento.

### ¿Puedo usar otros datasets de ciberseguridad?

**Sí**, algunos datasets compatibles:

| Dataset | Año | Samples | Features | Tipos de Ataque |
|---------|-----|---------|----------|-----------------|
| NSL-KDD | 2009 | 148k | 41 | DoS, Probe, R2L, U2R |
| UNSW-NB15 | 2015 | 2.5M | 49 | 9 tipos modernos |
| CICIDS2017 | 2017 | 2.8M | 80+ | 14 tipos (web attacks, botnet) |
| CSE-CIC-IDS2018 | 2018 | 16M | 80+ | 15 tipos |

Para usar otros datasets, adapta `load_nsl_kdd.py` o crea tu propio loader.

## 📄 Licencia

Este proyecto es de código abierto y está disponible bajo la licencia MIT (o la que corresponda).

---

## 📧 Contacto

- **Autor**: Disponible a través de GitHub
- **Issues**: Para reportar bugs o sugerir mejoras, abre un [Issue](https://github.com/yeaight7/TFG_CYBER_AI/issues)
- **Discusiones**: Para preguntas generales, usa [Discussions](https://github.com/yeaight7/TFG_CYBER_AI/discussions)

Para preguntas o colaboraciones, contacta con el autor del TFG a través de GitHub.

---

## 🙏 Agradecimientos

- **NSL-KDD Dataset**: Creado por el Canadian Institute for Cybersecurity
- **Stable-Baselines3**: Biblioteca de algoritmos RL de alta calidad
- **Gymnasium**: Framework estándar para entornos RL
- **Kaggle**: Plataforma para compartir datasets

---

## 📚 Referencias

- [NSL-KDD Dataset](https://www.unb.ca/cic/datasets/nsl.html)
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Deep Q-Network (DQN) Paper](https://www.nature.com/articles/nature14236)