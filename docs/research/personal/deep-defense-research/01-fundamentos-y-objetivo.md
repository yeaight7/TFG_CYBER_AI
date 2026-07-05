# 01 — Fundamentos y objetivo del TFG

## 1) Qué problema resuelve el proyecto

El sistema toma decisiones binarias sobre flujos de red:

- `0 = PERMIT`
- `1 = BLOCK`

La idea central es que en ciberseguridad los errores no cuestan lo mismo:

- **Falso negativo (FN)**: dejar pasar un ataque suele ser el error más grave.
- **Falso positivo (FP)**: bloquear tráfico benigno también duele, pero normalmente menos.

Por eso el proyecto se formula con **Aprendizaje por Refuerzo (RL)** en lugar de quedarse solo en clasificación supervisada clásica.

## 2) Objetivo real del trabajo

El objetivo no es solo “sacar accuracy alta”, sino construir un pipeline reproducible de extremo a extremo:

1. adaptar datos de red a un contrato común
2. entrenar un agente RL sobre ese contrato
3. validar con pruebas anti-autoengaño (A/B/C + leave-one-exact-CSV-out)
4. llevar el modelo a inferencia offline en tráfico real de laboratorio

## 3) Estructura por fases

### Fase 1 (más madura)

- entrenamiento y validación offline
- dataset principal: CICIDS2017
- baseline histórico: NSL-KDD

### Fase 2 (implementada pero abierta en robustez)

- inferencia offline sobre flujos extraídos de tráfico de laboratorio
- script mantenido: `scripts/predict_real_traffic_v2.py`
- foco actual: controlar el **domain shift** entre dataset y tráfico real

## 4) Qué está implementado y qué no

### Implementado

- esquema canónico fijo de 76 features
- máscara de missingness (otras 76)
- observación final de 152 dimensiones
- entorno RL custom con Gymnasium
- entrenamiento QRDQN
- validaciones A/B/C
- validación leave-one-exact-CSV-out en código
- inferencia robusta offline en Phase 2

### No implementado todavía

- bloqueo activo en tiempo real (`iptables`/`nftables`)
- cierre completo de calibración en tráfico benigno real
- despliegue productivo
- línea multiagente/adversarial

## 5) Mensaje fuerte para defensa

El valor del TFG está en combinar:

- rigor de ingeniería de datos (contrato de observación estable)
- diseño de decisión con costes asimétricos (reward shaping)
- validación que evita sobreinterpretar métricas cómodas
- transición realista hacia tráfico real, con límites reconocidos

