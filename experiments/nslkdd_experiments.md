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
____________________________________________________________________________________________________________________________________________________________________
| ID  | Modelo | Dataset                  | Reward (tp, fp, fn, om) | Steps   | Acc    | Rec atk | FP rate | Notas                                                 |
|-----|--------|--------------------------|-------------------------|---------|--------|---------|---------|-------------------------------------------------------|
| E01 | DQN    | NSL-KDD (20% train)      | 1.0, -1.0, -2.0, 0.0    |  200k   | 0.7602 | 0.600   | 0.028   | Baseline RL inicial (recompensa más suave en FN)      |
| E02 | RF     | NSL-KDD (20% train)      | -                       |   -     | 0.7693 | 0.615   | 0.0267  | Baseline supervisado Random Forest                    |
| E03 | DQN    | NSL-KDD (20% train)      | 1.0, -1.0, -5.0, 0.5    |  1000k  | 0.7208 | 0.528   | 0.0249  | RL con FN duro + omisión en benignos                  |
| E04 | DQN    | NSL-KDD (full train)     | 1.0, -1.0, -5.0, 0.5    |  1000k  | 0.7155 | 0.518   |  TODO   | Misma reward, entrenado con NSL-KDD completo          |
| E05 | DQN    | NSL-KDD (20% train)      | 2.0, -1.0, -6.0, 0.2    |  500k   | 0.7563 | 0.5955  | 0.0313  | Reward más agresiva pro-seguridad (FN muy penalizado) |
| E06 | DQN    | NSL-KDD (20% train)      | 1.5, -1.0, -5.0, 0.0    |  500k   | 0.7555 | 0.5928  | 0.0296  | Sin recompensa por omisión, ligera subida de FP       |
|_____|________|__________________________|_________________________|_________|________|_________|_________|_______________________________________________________|
```