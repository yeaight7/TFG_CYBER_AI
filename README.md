# TFG – Agente de Ciberseguridad con Aprendizaje por Refuerzo

Este repositorio contiene la primera fase de un Trabajo Fin de Grado orientado al diseño de un **agente defensor** basado en **Aprendizaje por Refuerzo (Reinforcement Learning, RL)** para tareas de detección y bloqueo de tráfico malicioso.

En esta fase el entorno es **simulado / tipo dataset**: el agente recibe características de flujos de red (u otras muestras etiquetadas como benignas o maliciosas) y aprende una política para **permitir o bloquear** el tráfico maximizando una función de recompensa.

---

## Estructura del proyecto

```text
tfg_cyber_ai/
├── src/
│   ├── rl_defender_env.py
│   └── train_rl_defender.py
├── datasets/
│   └── ** datasets públicos **
├── models/
├── venv/
└── README.md
```