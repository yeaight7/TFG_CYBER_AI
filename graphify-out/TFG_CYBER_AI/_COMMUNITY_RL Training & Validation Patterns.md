---
type: community
members: 31
---

# RL Training & Validation Patterns

**Members:** 31 nodes

## Members
- [[Classification-as-RL  Contextual Bandit Equivalence]] - rationale - docs/Personal Research/data-structure-and-canonical-schema-research-report.md
- [[DummyVecEnv + Monitor Wrappers]] - rationale - docs/Personal Research/data-structure-and-canonical-schema-research-report.md
- [[Episode Mechanics train shuffle  test deterministic]] - rationale - docs/Personal Research/data-structure-and-canonical-schema-research-report.md
- [[Optuna objective function for QRDQN tuning]] - code - src/tune_hparams.py
- [[QRDQN RL model (sb3_contrib)]] - code - src/train_rl_defender.py
- [[RLDatasetDefenderEnv gymnasium environment]] - code - src/rl_defender_env.py
- [[Reward Config (tpfpfnomission)]] - rationale - src/rl_defender_env.py
- [[_build_aggregate_results fold aggregation]] - code - src/validate_leave_one_csv_out.py
- [[_compute_reward]] - code - src/rl_defender_env.py
- [[_evaluate_f1 function in tune_hparams]] - code - src/tune_hparams.py
- [[baseline_random_forest main (3-sweep eval)]] - code - src/baseline_random_forest.py
- [[check_a_direct_eval (direct model evaluation)]] - code - src/validate_checks.py
- [[check_b_shuffled_labels (anti-leakage test)]] - code - src/validate_checks.py
- [[check_c_csv_split (realistic CSV-split eval)]] - code - src/validate_checks.py
- [[evaluate_model function in train_rl_defender]] - code - src/train_rl_defender.py
- [[evaluate_model_direct (batched leave-one-out eval)]] - code - src/validate_leave_one_csv_out.py
- [[evaluate_random_forest function]] - code - src/baseline_random_forest.py
- [[list_cicids2017_csv_files function]] - code - src/load_cicids2017.py
- [[load_cicids2017_binary function]] - code - src/load_cicids2017.py
- [[load_cicids2017_csv_split function]] - code - src/load_cicids2017.py
- [[load_cicids2017_exact_csv_split function]] - code - src/load_cicids2017.py
- [[load_cicids2017_split unified loader]] - code - src/load_cicids2017.py
- [[make_env_fn environment factory]] - code - src/train_rl_defender.py
- [[test_env_initialization]] - code - tests/test_rl_defender_env.py
- [[test_env_step_and_reset]] - code - tests/test_rl_defender_env.py
- [[test_reward_logic_tp_fp_tn_fn]] - code - tests/test_reward_config.py
- [[test_unknown_label_reward]] - code - tests/test_reward_config.py
- [[train_random_forest function]] - code - src/baseline_random_forest.py
- [[train_rl_defender main training pipeline]] - code - src/train_rl_defender.py
- [[tune_hparams main Optuna study]] - code - src/tune_hparams.py
- [[validate_leave_one_csv_out main pipeline]] - code - src/validate_leave_one_csv_out.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/RL_Training__Validation_Patterns
SORT file.name ASC
```

## Connections to other communities
- 6 edges to [[_COMMUNITY_RL Defender Core Design]]
- 2 edges to [[_COMMUNITY_Canonical Feature Schema]]

## Top bridge nodes
- [[RLDatasetDefenderEnv gymnasium environment]] - degree 19, connects to 1 community
- [[load_cicids2017_split unified loader]] - degree 5, connects to 1 community
- [[load_cicids2017_binary function]] - degree 4, connects to 1 community
- [[test_reward_logic_tp_fp_tn_fn]] - degree 3, connects to 1 community
- [[Classification-as-RL  Contextual Bandit Equivalence]] - degree 3, connects to 1 community