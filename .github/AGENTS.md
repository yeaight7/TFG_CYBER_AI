# Instrucciones para GitHub Copilot Coding Agent

Este documento contiene instrucciones específicas para el **GitHub Copilot Coding Agent** cuando trabaje de forma autónoma en este repositorio.

## 📖 Contexto Obligatorio

Antes de hacer **cualquier cambio**, debes leer estos documentos en orden:

1. **`.github/AGENT_CONTEXT.md`**: Estado del proyecto, decisiones de diseño, próximos pasos
2. **`.github/copilot-instructions.md`**: Convenciones de código y reglas generales

Estos documentos son la "verdad oficial" del proyecto. Si hay contradicción entre tu conocimiento general y estos documentos, **siempre prevalecen estos documentos**.

## ✅ Checklist Pre-Cambio

Antes de modificar código o crear archivos nuevos, verifica:

- [ ] He leído `.github/AGENT_CONTEXT.md` completamente
- [ ] Entiendo el esquema canónico de features (`FEATURES_CANON`) y por qué existe
- [ ] Si añado un nuevo dataset, crearé un adapter que mapee al esquema canónico
- [ ] Si modifico features, actualizaré TODOS los adapters para mantener consistencia
- [ ] NO incluiré features que causen data leakage (IPs, timestamps, Flow IDs)
- [ ] Usaré máscara de missingness (`m_i`) para features ausentes, no solo 0
- [ ] Los cambios están alineados con "Próximos pasos" en `.github/AGENT_CONTEXT.md`

## 🎯 Reglas Fundamentales

### 1. Esquema Canónico de Features

El proyecto usa un **esquema canónico fijo** para todas las features:

```python
# Concepto (ejemplo ilustrativo, la definición real está en .github/AGENT_CONTEXT.md)
FEATURES_CANON = [
    "flow_duration",
    "total_fwd_packets",
    "total_bwd_packets",
    # ... más features basadas en flows
]

# Cada dataset debe adaptarse a este esquema
# Si una feature no existe en el dataset origen:
# 1. Se imputa con valor razonable (ej. 0, media, mediana)
# 2. Se marca con m_i = 0 en la máscara de missingness

# Vector final de observación:
# obs = [x_1, ..., x_d, m_1, ..., m_d]
```

**NUNCA** entrenes un modelo con vectores de diferentes longitudes o donde las features significan cosas distintas.

### 2. Adapters de Datasets

Cada dataset tiene su propio loader/adapter:

- **`load_nsl_kdd.py`**: Adapter para NSL-KDD (benchmark histórico)
- **`load_cicids2017.py`**: Adapter para CICIDS2017 (dataset principal moderno)
- **Futuros**: `load_<nombre_dataset>.py` para cada nuevo dataset

**Todos los adapters deben**:
- Retornar el mismo formato de salida: `(X_train, y_train, X_test, y_test, scaler, feature_names)`
- Mapear columnas al esquema canónico
- Usar máscara de missingness para features ausentes
- Eliminar columnas de leakage (IPs, timestamps, Flow IDs)
- Aplicar one-hot encoding a categóricas si es necesario
- Retornar arrays NumPy `float32` para X y `int64` para y
- Etiqueta binaria: `0 = BENIGN`, `1 = ATTACK`

### 3. No Data Leakage

Features **PROHIBIDAS** (causarían data leakage):
- ❌ Direcciones IP (Source IP, Destination IP)
- ❌ Timestamps absolutos
- ❌ Flow IDs o cualquier identificador único
- ❌ Puertos específicos como features directas (solo si es necesario como estadística agregada)

Features **PERMITIDAS** (estadísticas de flujos, extraíbles de PCAP):
- ✅ Duración de flujo (flow_duration)
- ✅ Número de paquetes/bytes enviados/recibidos
- ✅ Tasas de errores, flags, ventanas TCP
- ✅ Estadísticas agregadas (mean, std, min, max de tamaños de paquetes, inter-arrival times, etc.)
- ✅ Flags TCP (SYN, ACK, FIN, etc.) como features binarias o contadores

### 4. Convenciones de Código

```python
# ✅ CORRECTO: Usar pathlib.Path
from pathlib import Path

dataset_dir = Path("datasets") / "cicids2017"
model_path = Path("models") / f"{RUN_ID}.zip"

# ❌ INCORRECTO: Strings hardcodeados
dataset_dir = "/home/user/datasets/cicids2017"  # ¡NO!

# ✅ CORRECTO: Generar RUN_ID con timestamp
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_ID = f"{EXP_ID}_dqn_lr1e-4_{timestamp}"

# ✅ CORRECTO: Guardar resultados en runs/<RUN_ID>/
results_dir = Path("runs") / RUN_ID
results_dir.mkdir(parents=True, exist_ok=True)

# ✅ CORRECTO: Type hints siempre
def load_dataset(path: Path, use_subset: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    ...

# ✅ CORRECTO: Usar if __name__ == "__main__":
if __name__ == "__main__":
    main()
```

### 5. Soporte Multi-Dataset

El proyecto está diseñado para trabajar con **múltiples datasets** (no solo 2):

- Cada dataset pasa por su propio adapter que mapea al esquema canónico
- El agente ve siempre el mismo espacio de observación (misma longitud, mismas features)
- Features ausentes se marcan con la máscara de missingness
- **NSL-KDD** es solo un benchmark histórico (Fase 1), NO forma parte del modelo final para simulación
- **CICIDS2017** es el dataset principal que define el esquema canónico
- Se espera añadir **más datasets** (especialmente de la familia CIC) en el futuro

### 6. Sistema de Recompensas RL

El entorno RL (`rl_defender_env.py`) usa un sistema de recompensas configurable:

```python
REWARD_CONFIG = {
    "tp": 1.5,     # True Positive: bloquear ataque
    "fp": -1.0,    # False Positive: bloquear tráfico benigno
    "fn": -5.0,    # False Negative: permitir ataque (¡muy malo!)
    "omission": 0.0,  # True Negative: permitir tráfico benigno
}
```

- Ajusta estos valores para controlar el trade-off entre seguridad (minimizar FN) y disponibilidad (minimizar FP)
- Penalizaciones más fuertes en FN → agente más agresivo bloqueando
- Penalizaciones más fuertes en FP → agente más conservador bloqueando

## 🚫 Qué NO Hacer

1. **NO modificar `docs/discusion_con_llm.md`**  
   Es un log histórico de conversaciones. Solo léelo para contexto, nunca lo modifiques. Puede que esté obsoleto o en desuso

2. **NO entrenar con diferentes vectores de features**  
   Todos los datasets deben pasar por el esquema canónico. Entrenar con vectores de longitud o significado diferente rompe el modelo.

3. **NO hardcodear paths absolutos**  
   Usa `pathlib.Path` con paths relativos desde la raíz del repo.

4. **NO eliminar runs antiguos sin justificación**  
   Los directorios en `runs/<RUN_ID>/` contienen experimentos históricos. No los borres a menos que sea estrictamente necesario.

5. **NO incluir features de leakage**  
   IPs, timestamps, Flow IDs están prohibidos como features del modelo.

6. **NO asumir solo 2 datasets**  
   El diseño debe soportar N datasets mediante adapters. Piensa siempre en extensibilidad.

7. **NO crear scripts sin RUN_ID**  
   Cualquier script de entrenamiento o experimentación debe generar un RUN_ID único y guardar resultados en `runs/<RUN_ID>/`.

8. **NO ignorar la máscara de missingness**  
   Si una feature no existe en un dataset, no pongas simplemente 0. Imputa + marca con `m_i = 0`.

## ✨ Pasos de Validación Antes de Abrir PR

Antes de finalizar tu trabajo y abrir un Pull Request, verifica:

### 1. Código

- [ ] Todos los archivos Python tienen type hints
- [ ] Uso `pathlib.Path` para rutas, no strings hardcodeados
- [ ] Scripts principales tienen `if __name__ == "__main__": main()`
- [ ] RUN_ID se genera automáticamente con timestamp
- [ ] Resultados se guardan en `runs/<RUN_ID>/`
- [ ] No hay imports innecesarios o código muerto
- [ ] Código sigue convenciones: `snake_case` para funciones/variables, `PascalCase` para clases

### 2. Datasets y Features

- [ ] Si añadí un nuevo dataset, creé un adapter que mapea al esquema canónico
- [ ] El adapter usa máscara de missingness para features ausentes
- [ ] No incluyo features de leakage (IPs, timestamps, Flow IDs)
- [ ] Etiquetas son binarias: `0 = BENIGN`, `1 = ATTACK`
- [ ] Arrays NumPy son `float32` (X) y `int64` (y)

### 3. Documentación

- [ ] Si cambié el esquema canónico, actualicé `.github/AGENT_CONTEXT.md`
- [ ] Si añadí un dataset, documenté su uso en `.github/AGENT_CONTEXT.md` sección "Datasets"
- [ ] Si hice un experimento, documenté resultados en `experiments/`
- [ ] Actualicé `README.md` si añadí nuevas funcionalidades visibles para el usuario

### 4. Testing (cuando exista infraestructura)

- [ ] Si hay tests, los ejecuté y pasan correctamente
- [ ] Si creé nuevas funciones críticas, consideré añadir tests unitarios

### 5. Reproducibilidad

- [ ] Usé seeds fijos (ej. `SEED = 42`) en experimentos
- [ ] Documenté hiperparámetros utilizados
- [ ] Guardé modelos entrenados con nombres descriptivos

## 🔄 Flujo de Trabajo Recomendado

1. **Lee contexto**: `.github/AGENT_CONTEXT.md` → `.github/copilot-instructions.md`
2. **Verifica alineación**: Asegúrate de que tu tarea está en "Próximos pasos" de `.github/AGENT_CONTEXT.md`
3. **Planifica**: Antes de codificar, piensa en cómo encaja con el esquema canónico y multi-dataset
4. **Implementa**: Escribe código siguiendo convenciones (type hints, pathlib, RUN_ID, etc.)
5. **Valida**: Ejecuta el código, verifica resultados, revisa checklist pre-PR
6. **Documenta**: Actualiza `.github/AGENT_CONTEXT.md` si cambiaste el estado del proyecto
7. **Abre PR**: Con descripción clara de qué cambios hiciste y por qué

## 📊 Estructura de Experimentos

Cuando crees experimentos, usa esta estructura:

```
runs/
└── <RUN_ID>/
    ├── config.json          # Configuración del experimento (hiperparámetros)
    ├── metrics.json         # Métricas finales (accuracy, precision, recall, F1)
    ├── confusion_matrix.png # Visualización opcional
    ├── logs/                # Logs de entrenamiento
    └── tensorboard/         # Logs de TensorBoard (si aplica)

experiments/
└── <nombre_experimento>.md  # Documentación del experimento, resultados, conclusiones
```

## 🎯 Ejemplo de Adapter de Dataset

Si añades un nuevo dataset, el adapter debe seguir este patrón:

```python
from pathlib import Path
from typing import Tuple, Optional, List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_<nombre_dataset>_binary(
    cfg: Optional[LoadConfig] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[StandardScaler], List[str]]:
    """
    Carga <nombre_dataset> y lo adapta al esquema canónico.
    
    Returns:
        X_train, y_train, X_test, y_test, scaler, feature_names
        
    - X: float32, shape (n_samples, n_features_canonical)
    - y: int64, shape (n_samples,), 0=BENIGN, 1=ATTACK
    - scaler: StandardScaler ajustado en train (o None si no se escaló)
    - feature_names: lista de nombres de features canónicas
    """
    cfg = cfg or LoadConfig()
    
    # 1. Cargar datos raw
    df = _load_raw_data(cfg)
    
    # 2. Mapear a esquema canónico + máscara de missingness
    X_canonical, missingness_mask = _map_to_canonical(df, FEATURES_CANON)
    
    # Combinar features + máscara
    X_combined = np.hstack([X_canonical, missingness_mask])
    
    # 3. Etiqueta binaria
    y = _extract_binary_labels(df, cfg.label_col, cfg.benign_value)
    
    # 4. Split estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, test_size=cfg.test_size, 
        random_state=cfg.random_state, stratify=y
    )
    
    # 5. Escalado (solo en train)
    scaler = None
    if cfg.scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_test = scaler.transform(X_test).astype(np.float32)
    
    feature_names = FEATURES_CANON + [f"m_{f}" for f in FEATURES_CANON]
    
    return X_train, y_train, X_test, y_test, scaler, feature_names
```

## 🧠 Preguntas Frecuentes

### ¿Qué es el esquema canónico?

Es una lista fija de features que TODOS los datasets deben tener. Define el espacio de observación del agente RL. Ejemplo: `["flow_duration", "total_fwd_packets", "total_bwd_packets", ...]`.

### ¿Qué hago si un dataset no tiene una feature canónica?

1. Imputa con valor razonable (0, media, mediana según contexto)
2. Marca con `m_i = 0` en la máscara de missingness
3. El agente aprenderá que esa feature no es confiable para ese dataset

### ¿Por qué NSL-KDD no es parte del modelo final?

NSL-KDD tiene features muy diferentes (antiguas, no basadas en flows modernos). Se usa como benchmark histórico en Fase 1 para demostrar el framework RL, pero no es compatible con el esquema canónico diseñado para CICIDS2017 y simulación.

### ¿Puedo cambiar el esquema canónico?

Sí, pero es un cambio mayor que requiere:
1. Actualizar `.github/AGENT_CONTEXT.md` con la nueva lista
2. Actualizar TODOS los adapters existentes para mapear al nuevo esquema
3. Re-entrenar todos los modelos
4. Justificar el cambio en la documentación

### ¿Cómo pruebo mi código si no hay tests?

Por ahora, validación manual:
1. Ejecuta el script y verifica que no hay errores
2. Revisa las métricas de salida (confusion matrix, classification report)
3. Verifica que los archivos se guardan en `runs/<RUN_ID>/` correctamente
4. Si es un loader, verifica shapes: `X.shape[1]` debe coincidir con número de features canónicas + máscara

## 📞 Si Tienes Dudas

1. Lee `.github/AGENT_CONTEXT.md` de nuevo
2. Si aún tienes dudas, documenta el problema en un Issue de GitHub y espera feedback del usuario

---

**Recuerda**: Este proyecto tiene un diseño cuidadoso para soportar múltiples datasets y transición a simulación. Respeta el esquema canónico, usa máscara de missingness, y mantén la trazabilidad con RUN_IDs. ¡Buena suerte! 🚀
