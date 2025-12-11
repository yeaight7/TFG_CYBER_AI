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
│   ├── train_rl_defender.py      # Script principal de entrenamiento
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
│   └── rl_defender_dqn.zip       # Modelo DQN entrenado (guardado)
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

NSL-KDD es un dataset de detección de intrusiones derivado del KDD Cup 1999. Contiene registros de conexiones de red con:

- **125,973 muestras de entrenamiento** (KDDTrain+.txt)
- **22,544 muestras de prueba** (KDDTest+.txt)
- **41 características numéricas/categóricas** por muestra
- **Etiquetas**: Normal, DoS, Probe, R2L, U2R

### Preprocesamiento Automático

El script `load_nsl_kdd.py` realiza:

1. **Descarga automática** desde Kaggle
2. **One-hot encoding** de variables categóricas (protocol_type, service, flag)
3. **Binarización de etiquetas**: 0 = Normal, 1 = Ataque
4. **División train/test** manteniendo la proporción original
5. **Conversión a arrays NumPy** (float32) para eficiencia

### Uso

```python
from load_nsl_kdd import load_nsl_kdd_binary

# Cargar dataset completo
X_train, y_train, X_test, y_test = load_nsl_kdd_binary(use_20_percent=False)

# O usar versión reducida (20%) para pruebas rápidas
X_train, y_train, X_test, y_test = load_nsl_kdd_binary(use_20_percent=True)
```

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
   - Total timesteps: 1,000,000
   - Cada episodio recorre hasta 10,000 muestras
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
Real Normal (0)        TP              FP
Real Ataque (1)        FN              TP
```

### Métricas Clave

- **Precision**: TP / (TP + FP) - Proporción de bloqueos correctos
- **Recall**: TP / (TP + FN) - Proporción de ataques detectados
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

---

## 🔬 Experimentación

### Probar Diferentes Algoritmos RL

Además de DQN, puedes experimentar con:

```python
from stable_baselines3 import PPO, A2C, SAC

# PPO (Proximal Policy Optimization)
model = PPO("MlpPolicy", env, verbose=1)

# A2C (Advantage Actor-Critic)
model = A2C("MlpPolicy", env, verbose=1)
```

### Ajustar Sistema de Recompensas

En `rl_defender_env.py`, modifica:

```python
RLDatasetDefenderEnv(
    X=X,
    y=y,
    correct_reward=2.0,              # Aumentar recompensa por aciertos
    false_positive_penalty=-0.5,     # Reducir penalización de FP
    false_negative_penalty=-10.0,    # Aumentar penalización de FN
    # ...
)
```

### Entrenar con Subset Reducido

Para experimentación rápida:

```python
# En train_rl_defender.py
X_train, y_train, X_test, y_test = load_nsl_kdd_binary(
    use_20_percent=True  # Solo 20% del dataset
)
```

---

## 📈 Cargar Modelo Pre-entrenado

Para evaluar o continuar el entrenamiento:

```python
from stable_baselines3 import DQN

# Cargar modelo guardado
model = DQN.load("models/rl_defender_dqn")

# Evaluar en nuevo conjunto
obs, info = env.reset()
action, _ = model.predict(obs, deterministic=True)
```

---

## 🔮 Trabajo Futuro

Esta es la **Fase 1** del TFG. Las siguientes fases incluirán:

### Fase 2: Entorno en Tiempo Real
- Integración con captura de tráfico en vivo (pcap, Wireshark)
- Uso de herramientas como Scapy para análisis de paquetes
- Pipeline de procesamiento en streaming

### Fase 3: Adversario RL
- Implementar un **agente atacante** también basado en RL
- Escenario de juego adversarial (Game Theory)
- Co-evolución defensor vs. atacante

### Fase 4: Multi-Agente
- Sistema distribuido con múltiples defensores
- Coordinación y comunicación entre agentes
- Defensa de redes complejas

### Fase 5: Despliegue
- Contenedorización (Docker)
- Integración con firewalls (iptables, nftables)
- Dashboard de monitorización

---

## 🤝 Contribuciones

Este es un proyecto académico (TFG), pero se aceptan sugerencias y mejoras:

1. Haz fork del repositorio
2. Crea una rama para tu feature (`git checkout -b feature/mejora`)
3. Commit tus cambios (`git commit -am 'Añadir mejora X'`)
4. Push a la rama (`git push origin feature/mejora`)
5. Abre un Pull Request

---

## 📄 Licencia

Este proyecto es de código abierto y está disponible bajo la licencia MIT (o la que corresponda).

---

## 📧 Contacto

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