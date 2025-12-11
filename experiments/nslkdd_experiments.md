# Experimentos NSL-KDD – DQN y Random Forest

Este fichero recoge los experimentos realizados sobre el dataset **NSL-KDD completo**
(KDDTrain+.TXT / KDDTest+.TXT) para comparar:

- El agente defensor basado en **DQN** (entorno RL).
- Un modelo supervisado clásico (**Random Forest**).

---

## Tabla de experimentos

**Leyenda columnas:**

- `ID`: identificador corto del experimento.
- `Modelo`: algoritmo principal (DQN, RF, etc.).
- `Dataset`: variante utilizada (NSL-KDD completo, 20%, etc.).
- `Reward (tp, fp, fn, om)`:
  - `tp`: recompensa por bloquear ataque.
  - `fp`: penalización por bloquear tráfico benigno.
  - `fn`: penalización por permitir ataque.
  - `om`: recompensa por permitir tráfico benigno (omisión).
- `Steps`: número de timesteps de entrenamiento (solo aplica a RL).
- `Acc`: accuracy global en test.
- `Rec atk`: recall de la clase ataque (clase 1).
- `FP rate`: tasa de falsos positivos (normal predicho como ataque / normales totales).
- `Notas`: comentario breve.

```markdown
| ID  | Modelo | Dataset          | Reward (tp, fp, fn, om) | Steps  | Acc    | Rec atk | FP rate| Notas                                  |
|-----|--------|------------------|-------------------------|--------|--------|---------|--------|-----------------------------------------|
| E01 | DQN    | NSL-KDD 		  | 1.0, -1.0, -2.0, 0.0    | 200k   | 0.7602 | 0.600   | 0.028  | Baseline RL inicial (sin omission)      |
| E02 | RF     | NSL-KDD 		  | -                       |   -    | TODO   | TODO    | TODO   | Random Forest baseline                  |
| E03 | DQN    | NSL-KDD 		  | 1.0, -1.0, -5.0, 0.5    | 200k   | TODO   | TODO    | TODO   | RL con FN alto + omisión en benignos    |