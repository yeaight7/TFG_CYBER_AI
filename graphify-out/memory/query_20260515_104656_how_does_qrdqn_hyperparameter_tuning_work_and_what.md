---
type: "query"
date: "2026-05-15T10:46:56.584721+00:00"
question: "How does QRDQN hyperparameter tuning work and what parameters does Optuna optimize?"
contributor: "graphify"
source_nodes: ["Optuna objective function for QRDQN tuning", "tune_hparams main Optuna study", "_evaluate_f1 function in tune_hparams", "QRDQN RL model (sb3_contrib)", "RLDatasetDefenderEnv gymnasium environment"]
---

# Q: How does QRDQN hyperparameter tuning work and what parameters does Optuna optimize?

## Answer

tune_hparams.py runs an Optuna study that optimizes QRDQN hyperparameters with F1 on the attack class as the objective. The pipeline: (1) main() loads CICIDS2017 via load_cicids2017_binary(), then calls Optuna's study.optimize(objective). (2) objective() receives an Optuna Trial, samples hyperparameters (learning_rate, n_quantiles, batch_size, gamma, exploration_fraction, etc.), creates an RLDatasetDefenderEnv, trains QRDQN for a short budget, then delegates to _evaluate_f1() for scoring. (3) _evaluate_f1() rolls out the trained policy on a held-out split and returns macro F1. The graph shows QRDQN RL model shares_data_with Optuna objective function (INFERRED, 0.85) which is accurate: both use the same sb3_contrib.QRDQN interface. Best found hyperparams are recorded in runs/cicids2017/ and the best run was C03 (accuracy=0.99859). The Deep Defense doc confirms Optuna found: n_quantiles sensitive to dataset, exploration_fraction around 0.1, Adam eps coupling matters for stability.

## Source Nodes

- Optuna objective function for QRDQN tuning
- tune_hparams main Optuna study
- _evaluate_f1 function in tune_hparams
- QRDQN RL model (sb3_contrib)
- RLDatasetDefenderEnv gymnasium environment