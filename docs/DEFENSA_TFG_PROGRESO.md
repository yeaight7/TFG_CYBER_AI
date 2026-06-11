# Defensa TFG — Progreso, mensajes clave y hechos verificables

**Autor**: Javier Rivero Iglesias  
**Objetivo del documento**: preparar una defensa técnica rigurosa, alineada con el estado actual del repositorio y con los artefactos reales almacenados en `runs/`.

---

## 1. Mensaje central del TFG

El TFG demuestra que un agente defensor basado en Aprendizaje por Refuerzo puede:

- observar tráfico de red representado como **flujos**
- decidir entre **PERMIT** y **BLOCK**
- optimizar explícitamente un sistema de costes asimétricos típico de ciberseguridad

La aportación principal no es solo entrenar un clasificador, sino diseñar un pipeline coherente que conecte:

- datasets históricos y modernos
- un **esquema canónico** reutilizable
- validación rigurosa
- transición hacia tráfico real en laboratorio privado

---

## 2. Estado actual del proyecto

### Implementado

- esquema canónico fijo de **76 features**
- máscara de missingness de otras **76 dimensiones**
- vector final de observación de **152 dimensiones**
- adapter para **CICIDS2017**
- adapter para **NSL-KDD** como benchmark histórico
- entorno RL custom con `Gymnasium`
- entrenamiento con **QRDQN**
- validaciones A, B y C
- validación **leave-one-exact-CSV-out** implementada en código
- pipeline robusto de inferencia Phase 2: `predict_real_traffic_v2.py`

### No implementado todavía

- bloqueo activo en tiempo real con `iptables` / `nftables`
- calibración cerrada para tráfico benigno real
- multi-agent RL
- entrenamiento multi-dataset combinado ya operacional

---

## 3. Invariantes técnicos que debes mantener en la exposición

### Esquema canónico

- `FEATURES_CANON` tiene **76 features flow-based**
- la observación final es siempre:

```text
[x1..x76, m1..m76]
```

- dimensión total: **152**

### Semántica de la máscara

- `m_i = 1` si la feature estaba presente o era válida
- `m_i = 0` si la feature fue imputada o no existía

### Política anti-leakage

No entran como features del modelo:

- IPs
- timestamps absolutos
- Flow IDs
- puertos usados como atajo hacia la etiqueta

---

## 4. Datasets y papel de cada uno

### NSL-KDD

Se usa como benchmark histórico de Fase 1.

Sirve para:

- validar el framework RL en una fase temprana
- comparar con un baseline supervisado

No sirve como base del modelo final porque:

- sus features no representan bien flujos modernos
- no definen el esquema canónico
- no son la referencia adecuada para la transición a tráfico real

### CICIDS2017

Es el dataset principal del proyecto.

Es clave porque:

- define el espacio de observación moderno
- usa features flow-based compatibles con extractores reales
- permite conectar el entrenamiento offline con la Fase 2

Además, el repositorio ahora reconoce explícitamente los **8 CSVs oficiales** de CICIDS2017 y soporta validación leave-one-exact-CSV-out sobre ellos.

---

## 5. Arquitectura del pipeline

El pipeline conceptual que debes transmitir es:

```text
dataset / flows extraídos
    -> limpieza y eliminación de leakage
    -> mapping al esquema canónico
    -> adición de missingness mask
    -> escalado con scaler ajustado en train
    -> entorno RL
    -> agente QRDQN
    -> decisión PERMIT/BLOCK
    -> evaluación y artefactos reproducibles
```

La idea fuerte aquí es que el sistema es **modular**:

- el dataset se cambia vía adapter
- el esquema de observación no cambia
- el entorno y el agente trabajan sobre el mismo contrato

---

## 6. Sistema de recompensas: cómo explicarlo bien

### Defaults actuales del código

Hoy, el repositorio está normalizado en:

```python
REWARD_CONFIG = {
    "tp": 1.5,
    "fp": -2.0,
    "fn": -5.0,
    "omission": 0.0,
}
```

Esto coincide con:

- `src/train_rl_defender.py`
- `src/validate_checks.py`
- `src/validate_leave_one_csv_out.py`
- `src/rl_defender_env.py`

### Importante para no decir algo incorrecto

El **mejor run histórico** (`C03_qrdqn_cicids2017_canonical_full_random_20260223_232439`) **no** usó el default actual, sino:

```python
tp = 1.5
fp = -2.0
fn = -5.0
omission = 0.0
```

Por tanto, en la defensa conviene distinguir siempre entre:

- **defaults actuales del código**
- **configuración concreta de un run histórico**

---

## 7. Resultados que sí puedes afirmar con respaldo

### Mejor run histórico en CICIDS2017

Artefacto:

- `runs/cicids2017/C03_qrdqn_cicids2017_canonical_full_random_20260223_232439/`

Métricas:

- accuracy: **0.99859**
- recall ataque: **0.99945**
- F1 ataque: **0.99876**

Mensaje útil en defensa:

- el pipeline alcanza rendimiento muy alto en CICIDS2017
- pero ese rendimiento no debe extrapolarse automáticamente a tráfico real

### Check A

Artefacto:

- `runs/validation/VAL_checks_A_20260212_235443/`

Resultado:

- accuracy: **0.9939**
- TP = 4772
- FP = 60
- FN = 1
- TN = 5167

Interpretación:

- las métricas son reproducibles sin depender de la lógica interna del entorno

### Check B

Artefacto:

- `runs/validation/VAL_checks_B_20260212_235736/`

Resultado:

- shuffled accuracy: **0.4773**
- baseline de clase mayoritaria: **0.5227**
- `leakage_detected = false`

Interpretación:

- no hay evidencia de data leakage en ese artefacto

### Check C

Artefacto:

- `runs/validation/VAL_checks_C_20260213_004847/`

Resultado:

- accuracy: **0.84135**
- F1 ataque: **0.62578**

Interpretación:

- es una validación mucho más dura
- mide generalización a particiones temporales / por CSV más realistas

### Leave-one-exact-CSV-out

Estado actual:

- está **implementado** en `src/validate_leave_one_csv_out.py`
- **no hay todavía un artefacto completo comprometido en `runs/validation/`**

Cómo contarlo:

- el pipeline ya soporta esta validación más fina por CSV exacto
- a día de hoy, el repositorio no incluye todavía el run completo consolidado

---

## 8. Fase 2: qué puedes defender con rigor

### Qué existe

- un laboratorio privado documentado
- un pipeline de inferencia offline robusto
- artefactos reales en `runs/phase2/`

### Qué no debes simplificar demasiado

La Fase 2 no tiene un único resultado estable. Hay artefactos con comportamientos distintos sobre `flows_benign.csv`.

Ejemplos reales:

- `P2v2_pred_20260224_004121`:
  - `block_rate = 1.0`
  - `allow_rate = 0.0`
- `P2v2_pred_20260408_230318`:
  - `block_rate = 0.0`
  - `allow_rate = 1.0`

Interpretación correcta:

- la inferencia Phase 2 es muy sensible a la configuración y al artefacto exacto
- por eso cualquier afirmación sobre tráfico real debe citar el `RUN_ID`
- esto refuerza el mensaje de que el **domain shift** sigue siendo un riesgo abierto

---

## 9. Qué decir sobre domain shift

El domain shift sigue siendo una de las principales limitaciones.

La lectura correcta es:

- CICIDS2017 permite entrenar muy bien en offline
- en laboratorio real aparecen distribuciones distintas
- por eso se introdujeron:
  - scaler persistido
  - percentile clipping
  - z-score clipping
  - diagnósticos de distribución

El argumento no es “la Fase 2 ya está cerrada”, sino:

- “la Fase 2 ya tiene pipeline técnico serio”
- “el problema abierto principal es la calibración / robustez frente a distribución real”

---

## 10. Posicionamiento honesto del trabajo

Una formulación sólida para la defensa es:

- el proyecto ya resuelve con rigor la parte de arquitectura, entrenamiento, validación y trazabilidad
- el proyecto ha dejado implementado el paso a un escenario de tráfico real
- la parte más madura del trabajo está en la definición del espacio de observación, la validación y la reproducibilidad
- la parte todavía abierta está en la robustez de la inferencia sobre tráfico real

Eso es mucho más creíble que presentar Phase 2 como un problema completamente resuelto.

---

## 11. Mensajes fuertes para el tribunal

### Mensaje técnico

No se ha entrenado “un clasificador cualquiera”, sino un sistema completo con:

- contrato de features estable
- prevención explícita de leakage
- separación entre defaults actuales y resultados históricos
- trazabilidad por `RUN_ID`

### Mensaje metodológico

El trabajo no se queda en accuracy sobre un dataset cómodo:

- hay validación directa
- hay prueba anti-leakage
- hay prueba de generalización dura
- hay transición a tráfico real

### Mensaje honesto

El problema real ya no es si el pipeline funciona, sino cuánto generaliza cuando sale del dataset hacia tráfico capturado en laboratorio.

---

## 12. Riesgos discursivos que conviene evitar

- No digas que Phase 2 ya está “resuelta”.
- No digas que el best model usa los mismos rewards que el código actual.
- No cites resultados leave-one-exact-CSV-out como si ya existiera un artefacto completo.
- No mezcles NSL-KDD como si formara parte del modelo final.

---

## 13. Cierre recomendado

Una formulación equilibrada para cerrar sería:

> El TFG deja una base experimental sólida y reproducible para un defensor RL sobre tráfico de red. La aportación principal está en el diseño coherente del pipeline: esquema canónico, adapters, entorno RL, validación rigurosa y transición operativa a tráfico real. El mejor rendimiento offline en CICIDS2017 es muy alto, y el trabajo deja además identificado con claridad el siguiente reto científico y práctico: la robustez frente a domain shift en la Fase 2.
