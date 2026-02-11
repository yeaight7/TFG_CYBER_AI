# GitHub Copilot Instructions — TFG_CYBER_AI

Este documento contiene instrucciones globales para GitHub Copilot cuando trabaje en este repositorio (chat, coding agent, code review).

## 📋 Descripción del Proyecto

**TFG_CYBER_AI** es un Trabajo de Fin de Grado sobre el desarrollo de un **agente defensor de ciberseguridad basado en Aprendizaje por Refuerzo (Reinforcement Learning)**. El agente aprende a clasificar tráfico de red como benigno o ataque, y decide acciones de PERMIT (permitir) o BLOCK (bloquear) para maximizar la seguridad minimizando falsos positivos y falsos negativos.

### Fases del Proyecto

- **Fase 1**: Clasificación y detección sobre datasets históricos y modernos (NSL-KDD como benchmark, CICIDS2017 como dataset principal, y potencialmente más datasets en el futuro).
- **Fase 2**: Entorno simulado con generación de tráfico real, extracción de características (feature extraction), y decisiones del agente en tiempo real sobre tráfico capturado.

## 🏗️ Estructura del Repositorio

```
TFG_CYBER_AI/
├── .github/              — Configuración de GitHub y documentación para Copilot
│   ├── AGENT_CONTEXT.md  — Contexto del proyecto y estado actual (LEER PRIMERO)
│   └── copilot-instructions.md  — Este archivo (instrucciones globales)
├── datasets/             — Datos (NSL-KDD, CICIDS2017, etc.)
├── docs/                 — Documentación del proyecto y decisiones de diseño
│   └── discusion_con_llm.md  — Log de conversaciones y decisiones (Es posible que esté obsoleto o en desuso) (NO MODIFICAR)
├── experiments/          — Tracking de experimentos, resultados, métricas
├── models/               — Modelos entrenados guardados (.zip, .joblib)
├── report/               — Memoria del TFG (LaTeX)
├── runs/                 — Resultados de ejecuciones con RUN_ID
│   └── <RUN_ID>/         — Carpeta por experimento (logs, métricas, TensorBoard)
├── src/                  — Código fuente Python principal
│   ├── baseline_random_forest.py  — Baseline supervisado con Random Forest
│   ├── load_cicids2017.py         — Loader para dataset CICIDS2017
│   ├── load_nsl_kdd.py            — Loader para dataset NSL-KDD
│   ├── rl_defender_env.py         — Entorno RL custom (Gymnasium)
│   └── train_rl_defender.py       — Script de entrenamiento DQN con logging
├── .gitignore
├── AGENTS.md             — Instrucciones específicas para coding agent
└── README.md             — Documentación general del proyecto
```

## 💻 Lenguaje y Convenciones de Código

### Lenguaje y Estilo

- **Lenguaje**: Python 3.10+ (preferiblemente 3.10 o 3.11)
- **Tipo de hints**: SIEMPRE usar type hints en funciones y métodos
- **Naming conventions**:
  - Variables y funciones: `snake_case`
  - Clases: `PascalCase`
  - Constantes: `UPPER_SNAKE_CASE`
- **Imports**: Ordenados alfabéticamente, con imports de librerías estándar primero, luego third-party, luego locales
- **Docstrings**: Usar docstrings en funciones no triviales, estilo Google/NumPy

### Patrones de Código Estándar

1. **Scripts ejecutables**: Todos los scripts principales deben usar el patrón:
   ```python
   if __name__ == "__main__":
       main()
   ```

2. **Paths**: Usar `pathlib.Path` en lugar de strings para rutas de archivos:
   ```python
   from pathlib import Path
   
   dataset_dir = Path("datasets") / "nsl_kdd"
   ```

3. **RUN_ID y logging**: Los experimentos deben generar un RUN_ID único con timestamp:
   ```python
   from datetime import datetime
   
   timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
   RUN_ID = f"{EXP_ID}_dqn_lr1e-4_{timestamp}"
   ```

4. **Resultados**: Los resultados de experimentos se guardan en `runs/<RUN_ID>/`
   - Nunca usar paths hardcodeados
   - Crear directorios con `mkdir(parents=True, exist_ok=True)`

5. **Seeds**: Para reproducibilidad, usar `random_state` o `seed` con valor por defecto (ej. 42):
   ```python
   SEED = 42
   np.random.seed(SEED)
   model = DQN(..., seed=SEED)
   ```

## ⚠️ Reglas Críticas de Diseño

### 1. Esquema Canónico de Features (FEATURES_CANON)

**IMPORTANTE**: El proyecto usa un **esquema canónico de features** fijo para todos los datasets y para la simulación.

- **Todos los datasets** deben convertirse al mismo esquema canónico mediante **adapters**
- Las features canónicas son aquellas que:
  1. Existen o pueden extraerse de CICIDS2017 (dataset principal moderno)
  2. Pueden extraerse de tráfico real/PCAP mediante flow extractors (CICFlowMeter, Zeek, etc.)
  3. NO incluyen identificadores o información que cause data leakage (IPs, timestamps, Flow ID, puertos específicos como features directas)
  4. Son preferiblemente numéricas y representan estadísticas de flujos de red

### 2. Máscara de Missingness (Missingness Mask)

**NUNCA** pongas simplemente 0 para features que no existen en un dataset.

- Cuando una feature canónica no existe en un dataset origen, debe:
  1. **Imputarse** con un valor razonable (media, mediana, 0 según contexto)
  2. **Marcarse** con una máscara de missingness `m_i = 0` (0 = ausente, 1 = presente)

- El vector de observación final del agente será:
  ```
  obs = [x_1, x_2, ..., x_d, m_1, m_2, ..., m_d]
  ```
  donde `x_i` es el valor de la feature (imputado si falta) y `m_i` indica si estaba presente.

- Esto permite al agente RL aprender qué features son confiables y cuáles no.

### 3. No Data Leakage

**NUNCA** incluir en el vector de features información que cause leakage:

- ❌ Direcciones IP (source, destination)
- ❌ Timestamps absolutos
- ❌ Flow IDs o identificadores únicos
- ❌ Puertos específicos como features directas (pueden usarse para calcular estadísticas, pero no incluir como features brutas)

### 4. Soporte Multi-Dataset

El diseño debe soportar entrenar con **N datasets** (no solo 2):

- Cada dataset pasa por un **adapter** que lo convierte al esquema canónico
- El adapter debe:
  - Mapear columnas existentes a nombres canónicos
  - Calcular features derivadas si es necesario
  - Marcar features ausentes con la máscara de missingness
- Los adapters deben estar en archivos separados (ej. `load_nsl_kdd.py`, `load_cicids2017.py`, etc.)

**Datasets actuales**:
- **NSL-KDD**: Benchmark histórico (Fase 1), features antiguas. NO es parte del modelo final para simulación.
- **CICIDS2017**: Dataset principal moderno, base para definir FEATURES_CANON.
- **Futuros**: Se espera añadir más datasets de la familia CIC u otros compatibles mediante adapters.

## 🚀 Cómo Ejecutar Scripts Existentes

### Baseline Random Forest
```bash
cd src
python baseline_random_forest.py
```

### Entrenamiento RL (DQN)
```bash
cd src
python train_rl_defender.py
```
- Ajusta parámetros en el script (learning_rate, buffer_size, total_timesteps, etc.)
- Modifica `use_20_percent=True/False` en `load_nsl_kdd_binary()` para usar dataset completo o reducido
- El modelo se guarda en `models/<RUN_ID>.zip`
- Los logs de TensorBoard se guardan en `runs/nslkdd/<RUN_ID>/`

### Ver logs de TensorBoard
```bash
tensorboard --logdir runs/nslkdd
```

## 🧪 Testing

**NOTA**: El proyecto aún no tiene infraestructura de tests automatizados.

- Tests unitarios están pendientes de implementación
- Por ahora, validación manual mediante scripts de evaluación (evaluación en test set, confusion matrix, classification report)

## 📚 Documentos Clave a Leer

Antes de hacer cambios significativos, **SIEMPRE** lee:

1. **`.github/AGENT_CONTEXT.md`**: Contexto del proyecto, estado actual, decisiones de diseño, próximos pasos
2. **`AGENTS.md`**: Checklist específico para coding agents (si trabajas como coding agent autónomo)

## 🔒 Qué NO Hacer

- ❌ NO modificar `docs/discusion_con_llm.md` (es un log histórico de conversaciones con un agente)
- ❌ NO entrenar modelos con vectores de diferentes longitudes o significado de features
- ❌ NO hardcodear paths absolutos (usar `Path` relativas desde repo root)
- ❌ NO eliminar experimentos antiguos de `runs/` sin justificación
- ❌ NO incluir features que causen data leakage (IPs, timestamps, Flow IDs)
- ❌ NO asumir que solo hay 2 datasets; el diseño debe soportar N datasets

## 🎯 Objetivo del Agente RL

El agente defensor aprende una política para:
- **Acción 0 (PERMIT)**: Permitir el tráfico
- **Acción 1 (BLOCK)**: Bloquear el tráfico

**Sistema de recompensas** (configurable en `train_rl_defender.py`):
- **TP (True Positive)**: Bloquear ataque correctamente → recompensa positiva (+1.5 por defecto)
- **FP (False Positive)**: Bloquear tráfico benigno → penalización (-1.0 por defecto)
- **FN (False Negative)**: Permitir ataque → penalización fuerte (-5.0 por defecto)
- **TN (True Negative)**: Permitir tráfico benigno → recompensa menor (0.0 por defecto, configurable como "omission")

El balance entre FP y FN se controla ajustando las recompensas.

## 🔄 Pipeline de Trabajo Recomendado

1. Lee `.github/AGENT_CONTEXT.md` para entender el estado actual
2. Si trabajas en un issue/tarea, verifica que está alineado con "Próximos pasos" en `.github/AGENT_CONTEXT.md`
3. Escribe código siguiendo las convenciones (type hints, snake_case, pathlib, RUN_ID)
4. Si añades un nuevo dataset, crea un adapter que mapee al esquema canónico con máscara de missingness
5. Si modificas el esquema canónico, actualiza TODOS los adapters para mantener consistencia
6. Documenta experimentos en `experiments/` con resultados, hiperparámetros, y conclusiones
7. Usa commits descriptivos y mensajes informativos

## 📞 Contacto y Referencias

- **Repositorio**: https://github.com/yeaight7/TFG_CYBER_AI
- **Documentación adicional**: Ver `README.md` para instrucciones de instalación y uso
- **Referencias técnicas**: Ver `report/` para la memoria del TFG (LaTeX)
