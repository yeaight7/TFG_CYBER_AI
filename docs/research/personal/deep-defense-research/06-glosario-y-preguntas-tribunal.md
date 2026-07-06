# 06 — Glosario esencial + preguntas de tribunal

## 1) Glosario mínimo para explicar el proyecto a principiantes

- **Flujo de red**: resumen estadístico de una comunicación (no payload completo).
- **Feature**: variable numérica usada por el modelo para decidir.
- **Etiqueta (`y`)**: clase real (`0=BENIGN`, `1=ATTACK`).
- **Observación**: vector que recibe el agente (aquí 152 dimensiones).
- **Política**: regla aprendida para elegir acción.
- **Recompensa**: señal numérica que define qué decisiones son mejores.
- **FP/FN**:
  - FP: bloquear benigno por error
  - FN: permitir ataque por error
- **Recall de ataque**: proporción de ataques detectados.
- **F1 de ataque**: equilibrio entre precisión y recall de ataque.
- **Domain shift**: diferencia entre distribución de entrenamiento y tráfico real.
- **Scaler**: normalización estadística de features.
- **RUN_ID**: identificador único de una ejecución reproducible.

## 2) Preguntas frecuentes del tribunal (con respuesta técnica breve)

## “¿Por qué RL y no solo clasificación supervisada?”

Porque el proyecto necesita optimizar costes asimétricos de seguridad de forma explícita en la función de recompensa, no solo exactitud global.

## “¿Qué aporta el esquema canónico?”

Permite un contrato de observación estable entre datasets y Phase 2 real. Sin eso, el agente vería vectores incompatibles entre entrenamientos.

## “¿Qué significa realmente 152?”

76 features canónicas + 76 valores de máscara de missingness.

## “¿Cómo evitas leakage?”

Excluyendo IPs, timestamps, Flow IDs y puertos-proxy en el preprocesado; además Check B comprueba que el rendimiento colapsa al barajar etiquetas.

## “¿Tu mejor accuracy implica robustez real?”

No necesariamente. El random split da un techo in-distribution; la validación por día/CSV muestra dificultad de generalización más realista.

## “¿Está lista Phase 2 para producción?”

No. Está lista para inferencia offline robusta y diagnóstico; bloqueo activo y robustez operativa cerrada aún no.

## “¿Cuál es tu principal limitación actual?”

La sensibilidad al domain shift en tráfico real.

## 3) Estructura de explicación en 3 minutos (versión corta)

1. problema: decisión PERMIT/BLOCK con costes asimétricos
2. solución: contrato canónico (76+76), entorno RL, QRDQN, validación rigurosa
3. resultado: alto rendimiento offline + identificación honesta de límite real (domain shift en Phase 2)

## 4) Estructura de explicación en 10–12 minutos (versión defensa)

1. motivación y objetivos
2. datasets y esquema canónico
3. entorno RL, recompensa y entrenamiento
4. validaciones A/B/C + leave-one-CSV-out
5. Phase 2 robusta y riesgos abiertos
6. conclusiones, límites y próximos pasos

