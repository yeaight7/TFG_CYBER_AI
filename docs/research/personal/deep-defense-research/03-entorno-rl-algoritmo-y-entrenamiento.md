# 03 — Entorno RL, algoritmo y entrenamiento

> **⚠️ Alineamiento con el experimento oficial (MAIN) — leer primero.**
> Esta es una **nota de investigación**, no la fuente de verdad de la configuración. La configuración **oficial** es la del run **MAIN** (`MAIN_qrdqn_cicids2017_canonical_full_random_20260609_193655`, perfil `main-experiment`): `gamma=0.0`, `net_arch=[1024,1024,512]`, `n_quantiles=200` (explícito), `learning_rate=5e-5`, `batch_size=2048`, `exploration_fraction=0.10`, `exploration_final_eps=0.02`, `gradient_steps=20`, `train_freq=100`, `target_update_interval=10_000`, `buffer_size=1_000_000`, `learning_starts=50_000`, `max_grad_norm=10.0`, `timesteps=3_000_000`.
> Cualquier mención a `net_arch=[512,256]`, `gamma=0.99`, `learning_rate=1e-4` o `exploration_fraction=0.005` corresponde a **exploración previa** o al perfil **`default`** (dev/smoke), **no** al experimento oficial. Fuente de verdad: `src/train_rl_defender.py` (`resolve_training_hyperparams`, `REWARD_CONFIG`) y `runs/cicids2017/MAIN_.../config.json`.

## 1) Entorno RL: qué modela exactamente

Archivo: `src/rl_defender_env.py`

- observación: `X[i]`
- acción: `0=PERMIT`, `1=BLOCK`
- recompensa según etiqueta real y acción

Es una formulación de decisión secuencial simple sobre muestras etiquetadas (muy cercana a contextual bandit con coste asimétrico).

## 2) Recompensa actual por defecto

En código mantenido (`train_rl_defender.py`, `validate_checks.py`, `validate_leave_one_csv_out.py`, `rl_defender_env.py`):

```python
{
  "tp": 1.5,
  "fp": -2.0,
  "fn": -5.0,
  "omission": 0.0,
}
```

Interpretación:

- bloquear ataque recompensa moderadamente
- bloquear benigno penaliza
- permitir ataque penaliza fuertemente
- permitir benigno queda neutro (`omission`)

## 3) Sobre el algoritmo principal

Archivo: `src/train_rl_defender.py`

- se usa `QRDQN` (`sb3_contrib`)
- política: `MlpPolicy`
- arquitectura de red del experimento oficial (perfil `main-experiment`, run MAIN): `[1024, 1024, 512]` (el perfil `default` dev/smoke usa `[512, 256]`)
- entorno envuelto con `DummyVecEnv` y `Monitor`

## 4) Hiperparámetros relevantes en el experimento oficial (perfil `main-experiment`, run MAIN)

- `learning_rate = 5e-5`
- `gamma = 0.0`
- `tau = 1.0`
- `batch_size = 2048` (fijo)
- `gradient_steps = 20`
- `train_freq = 100`
- `target_update_interval = 10_000`
- `buffer_size = 1_000_000`, `learning_starts = 50_000`
- `exploration_fraction = 0.10`, `exploration_final_eps = 0.02` (fijados explícitamente)
- `max_grad_norm = 10.0`

> El perfil `default` (dev/smoke, **no oficial**) usa otros valores: `learning_rate=1e-4`, `gamma=0.0`, `exploration_fraction=0.005`, y valores dependientes de preset — `batch_size` `512` (fast) / `2048` (full), `gradient_steps` `10`/`20`, `train_freq` `50`/`100`, `target_update_interval` `1_000`/`10_000`.

## 5) Presets y timesteps

Código efectivo:

- experimento oficial (perfil `main-experiment`): `3_000_000` timesteps (run MAIN)
- perfil `default`: `25_000` (`fast`) / `100_000` (`full`) timesteps por defecto

Nota: el texto de ayuda de CLI sobre timesteps puede quedar desactualizado frente al valor real aplicado en `main()`.

## 6) Artefactos que deja cada entrenamiento

- modelo: `models/<RUN_ID>.zip`
- en `runs/cicids2017/<RUN_ID>/`:
  - `config.json`
  - `metrics.json`
  - `scaler.joblib`
  - `train_percentiles.npz`

Esto permite reproducibilidad y paso consistente a Phase 2.

## 7) Baseline y tuning (complemento)

## Baseline Random Forest

Archivo: `src/baseline_random_forest.py`

Sirve para comparar contra un modelo supervisado clásico.

## Tuning con Optuna

Archivo: `src/tune_hparams.py`

Explora hiperparámetros de QRDQN y optimiza F1 de ataque.

