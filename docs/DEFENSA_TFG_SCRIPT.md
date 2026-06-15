# Defensa TFG — Guion oral actualizado

**Autor**: Javier Rivero Iglesias
**Idioma**: español
**Objetivo**: guion de presentación oral alineado con el estado real del repositorio.

---

## 1. Apertura

Buenos días. Mi Trabajo de Fin de Grado se centra en el diseño de un **agente defensor de ciberseguridad basado en Aprendizaje por Refuerzo**, capaz de decidir si un flujo de red debe **permitirse o bloquearse**.

La idea de fondo es sencilla, pero importante: en ciberseguridad los costes no son simétricos. No cuesta lo mismo bloquear por error tráfico legítimo que dejar pasar un ataque real. Por eso elegí Aprendizaje por Refuerzo: porque permite codificar explícitamente estos costes en un sistema de recompensas configurable.

---

## 2. Objetivo del proyecto

El objetivo no era entrenar solo un clasificador con buena accuracy, sino construir un pipeline completo y reproducible que conectara:

- datasets de entrenamiento
- un espacio de observación coherente
- un entorno RL
- validación rigurosa
- y una transición hacia tráfico real en un laboratorio privado

---

## 3. Estructura por fases

El proyecto está dividido en dos fases:

- **Fase 1**: entrenamiento y validación offline sobre datasets
- **Fase 2**: inferencia offline sobre tráfico capturado en laboratorio

La Fase 1 está más madura y cerrada. La Fase 2 está técnicamente implementada, pero aún abierta en cuanto a robustez frente a tráfico real.

---

## 4. Datasets usados

He trabajado con dos datasets principales.

El primero es **NSL-KDD**, que utilicé como benchmark histórico de Fase 1. Fue útil para validar que el framework RL funcionaba, pero no sirve como base del modelo final porque sus features son antiguas y no están alineadas con extractores de flujo modernos.

El segundo, y realmente central, es **CICIDS2017**. Este dataset define el espacio de observación moderno del proyecto, porque sus features son flow-based y son compatibles con la transición a tráfico real.

---

## 5. Aportación técnica clave: el esquema canónico

Una de las decisiones más importantes del proyecto fue definir un **esquema canónico fijo de 76 features**.

Esto resuelve un problema fundamental: un agente RL necesita que cada posición del vector de observación tenga siempre el mismo significado.

Por tanto:

- no podemos mezclar datasets con features arbitrarias
- todos los datasets deben mapearse al mismo esquema
- el agente siempre ve un espacio consistente

---

## 6. Máscara de missingness

Además de las 76 features, añadí una **máscara de missingness** de otras 76 dimensiones.

Así, la observación final tiene **152 dimensiones**:

```text
[x1..x76, m1..m76]
```

Donde:

- `x_i` es el valor de la feature
- `m_i = 1` si esa feature estaba presente
- `m_i = 0` si hubo que imputarla

Esto permite entrenar con datasets heterogéneos sin romper el espacio de observación.

---

## 7. Prevención de data leakage

Otro punto central del proyecto fue la prevención explícita de **data leakage**.

El modelo no recibe como features:

- IPs
- timestamps absolutos
- Flow IDs
- puertos usados como atajo hacia la etiqueta

La idea es obligar al agente a aprender patrones reales de flujo, no shortcuts triviales.

---

## 8. Entorno RL y acciones

Implementé un entorno custom con `Gymnasium`.

En cada paso:

- el agente observa un flujo
- decide entre dos acciones:
  - `0 = PERMIT`
  - `1 = BLOCK`
- y recibe una recompensa según si acertó o no

Esto modela directamente la lógica de un defensor de red.

---

## 9. Sistema de recompensas

El sistema de recompensas actual del código está normalizado en:

```python
tp = 1.5
fp = -2.0
fn = -5.0
omission = 0.0
```

Es importante señalar que el **mejor run histórico**, el C03 full, usa esos mismos valores. Otros runs históricos anteriores, como C01/C02, sí usaron configuraciones distintas.

Esto conviene decirlo con claridad para no mezclar:

- defaults actuales del código
- configuración concreta de un artefacto histórico

---

## 10. Algoritmo principal

La versión madura del pipeline usa **QRDQN**, una variante distributional de DQN.

La ventaja es que no solo estima un valor esperado simple, sino una distribución del retorno, algo especialmente interesante cuando el problema tiene costes asimétricos y riesgo operacional.

---

## 11. Resultados principales en CICIDS2017

El mejor artefacto comprometido en el repositorio es:

- `C03_qrdqn_cicids2017_canonical_full_random_20260223_232439`

Con resultados de:

- accuracy: **0.99859**
- recall de ataque: **0.99945** (sensitivity)
- F1 de ataque: **0.99876**

Es decir, en el dataset offline el rendimiento es extremadamente alto.

---

## 12. Validación rigurosa

Para no quedarme solo con una accuracy bonita, implementé varias validaciones.

### Check A

Evalúa directamente `model.predict(X_test)` contra `y_test`, sin depender del entorno.

Resultado histórico:

- accuracy: **0.9939**

### Check B

Baraja las etiquetas del entrenamiento y re-entrena brevemente.

Resultado:

- shuffled accuracy: **0.4773**
- baseline: **0.5227**
- sin evidencia de leakage

### Check C

Hace una partición dura por CSV/día.

Resultado histórico:

- accuracy: **0.84135**

Este check es importante porque mide generalización en un escenario mucho más exigente.

---

## 13. Nueva validación leave-one-exact-CSV-out

Además, el repositorio ya incluye una validación más fina:

- `src/validate_leave_one_csv_out.py`

Esta validación deja exactamente un CSV real de CICIDS2017 para test en cada fold y entrena con los otros siete.

Lo importante aquí es ser preciso:

- la validación está **implementada**
- pero el repositorio **todavía no incluye un run completo comprometido** de este barrido

Por tanto, puedo presentarla como capacidad técnica del pipeline, no como una batería ya cerrada con métricas consolidadas en `runs/validation/`.

---

## 14. Fase 2: tráfico real en laboratorio

La Fase 2 lleva el pipeline a tráfico real capturado en un laboratorio privado.

Aquí el flujo es:

- generar tráfico benigno y malicioso
- capturar PCAPs
- extraer flujos
- mapearlos al esquema canónico
- ejecutar inferencia offline

El script mantenido para esto es:

- `scripts/predict_real_traffic_v2.py`

---

## 15. Qué ocurrió en la Fase 2

La conclusión importante de Phase 2 no es que el problema esté completamente resuelto, sino que el pipeline ya detectó un reto real: el **domain shift**.

Hay artefactos comprometidos en el repositorio con comportamientos distintos sobre tráfico benigno real.

Por ejemplo:

- `P2v2_pred_20260224_004121`: bloquea el 100 % de `flows_benign.csv`
- `P2v2_pred_20260408_230318`: permite el 100 % de `flows_benign.csv`

Esto demuestra que la inferencia real es sensible a la configuración exacta y que cualquier afirmación sobre tráfico real debe citar el `RUN_ID`.

---

## 16. Qué aporta esto científicamente

Creo que el valor del trabajo está en cuatro aportaciones claras:

1. un esquema canónico coherente y reusable
2. una arquitectura RL modular y reproducible
3. una validación más rigurosa que la típica accuracy aislada
4. una transición realista hacia tráfico real con identificación explícita del domain shift

---

## 17. Limitaciones

Las limitaciones principales hoy son:

- la robustez frente a tráfico benigno real todavía no está cerrada
- no hay bloqueo activo en tiempo real
- la validación leave-one-exact-CSV-out está implementada, pero falta el artefacto completo comprometido

---

## 18. Trabajo futuro

Las siguientes líneas naturales serían:

- ejecutar y consolidar la validación leave-one-exact-CSV-out
- calibrar o hacer fine-tuning para tráfico del laboratorio
- explorar bloqueo activo controlado
- incorporar más datasets modernos

---

## 19. Cierre

En resumen, el TFG deja una base experimental sólida y reproducible para un defensor RL sobre tráfico de red.

La parte mejor resuelta es la arquitectura técnica: esquema canónico, adapters, entorno RL, validación y trazabilidad.

La parte todavía abierta, y por tanto más interesante como trabajo futuro, es la robustez al pasar del dataset a tráfico real.

Muchas gracias. Quedo a disposición para las preguntas del tribunal.
