---
type: community
cohesion: 0.22
members: 11
---

# Hyperparams & Rewards

**Cohesion:** 0.22 - loosely connected
**Members:** 11 nodes

## Members
- [[Optuna objective function for QRDQN tuning]] - code - src/tune_hparams.py
- [[RLDatasetDefenderEnv gymnasium environment]] - code - src/rl_defender_env.py
- [[Reward Config (tpfpfnomission)]] - rationale - src/rl_defender_env.py
- [[Reward Config Schema (tpfpfnomission)]] - rationale - tests/test_reward_config.py
- [[_compute_reward]] - code - src/rl_defender_env.py
- [[_evaluate_f1 function in tune_hparams]] - code - src/tune_hparams.py
- [[test_env_initialization]] - code - tests/test_rl_defender_env.py
- [[test_env_step_and_reset]] - code - tests/test_rl_defender_env.py
- [[test_reward_logic_tp_fp_tn_fn]] - code - tests/test_reward_config.py
- [[test_unknown_label_reward]] - code - tests/test_reward_config.py
- [[tune_hparams main Optuna study]] - code - src/tune_hparams.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Hyperparams__Rewards
SORT file.name ASC
```
