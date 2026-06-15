---
type: community
cohesion: 0.08
members: 37
---

# Test Split Validation

**Cohesion:** 0.08 - loosely connected
**Members:** 37 nodes

## Members
- [[.__init__()_1]] - code - src/validate_checks.py
- [[._on_step()]] - code - src/validate_checks.py
- [[Baraja y_train y entrena brevemente. Si el modelo aún obtiene accuracy     alta]] - rationale - src/validate_checks.py
- [[BaseCallback]] - code
- [[Callback para mostrar progreso cada log_freq timesteps.]] - rationale - src/validate_checks.py
- [[Deterministic, stratified, nested train subsample indices.      Per-class se]] - rationale - src/load_cicids2017.py
- [[Entrena en unos CSVs de CICIDS2017 y testea en otros.     Es el split más reali]] - rationale - src/validate_checks.py
- [[Evaluación directa model.predict(X_testi) vs y_testi.     No pasa por el e]] - rationale - src/validate_checks.py
- [[ProgressCallback]] - code - src/validate_checks.py
- [[SHA-256 content hash of an ndarray, prefixed with dtype and shape so that     a]] - rationale - src/load_cicids2017.py
- [[Unified CICIDS2017 loader with split-mode and preset support.      Parameters]] - rationale - src/load_cicids2017.py
- [[_sha256_of_array()]] - code - src/load_cicids2017.py
- [[_split()]] - code - tests/test_load_cicids2017.py
- [[_stratified_nested_prefix_indices()]] - code - src/load_cicids2017.py
- [[_synthetic_df()]] - code - tests/test_load_cicids2017.py
- [[check_a_direct_eval()]] - code - src/validate_checks.py
- [[check_b_shuffled_labels()]] - code - src/validate_checks.py
- [[check_c_csv_split()]] - code - src/validate_checks.py
- [[fail()]] - code - scripts/verify_fixed_test_split.py
- [[load_cicids2017_split()]] - code - src/load_cicids2017.py
- [[main()_4]] - code - scripts/verify_fixed_test_split.py
- [[main()_8]] - code - src/validate_checks.py
- [[ok()]] - code - scripts/verify_fixed_test_split.py
- [[parse_args()_2]] - code - scripts/verify_fixed_test_split.py
- [[parse_args()_5]] - code - src/validate_checks.py
- [[patched_loader()]] - code - tests/test_load_cicids2017.py
- [[test_load_cicids2017.py]] - code - tests/test_load_cicids2017.py
- [[test_metadata_new_keys()]] - code - tests/test_load_cicids2017.py
- [[test_nested_prefix_indices_deterministic()]] - code - tests/test_load_cicids2017.py
- [[test_nested_prefix_indices_nested_and_stratified()]] - code - tests/test_load_cicids2017.py
- [[test_scale_true_refits_on_subsample()]] - code - tests/test_load_cicids2017.py
- [[test_sha256_of_array_stable()]] - code - tests/test_load_cicids2017.py
- [[test_train_max_rows_guards()]] - code - tests/test_load_cicids2017.py
- [[test_train_max_rows_keeps_test_set_identical()]] - code - tests/test_load_cicids2017.py
- [[validate_checks.py]] - code - src/validate_checks.py
- [[validate_checks.py — Validación de resultados experimentales del agente RL.  I]] - rationale - src/validate_checks.py
- [[verify_fixed_test_split.py]] - code - scripts/verify_fixed_test_split.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Test_Split_Validation
SORT file.name ASC
```

## Connections to other communities
- 7 edges to [[_COMMUNITY_CICIDS2017 Preprocessing]]
- 1 edge to [[_COMMUNITY_RL Defender Training]]
- 1 edge to [[_COMMUNITY_RL Environment Config]]
- 1 edge to [[_COMMUNITY_Project Overview]]

## Top bridge nodes
- [[load_cicids2017_split()]] - degree 12, connects to 2 communities
- [[test_load_cicids2017.py]] - degree 11, connects to 1 community
- [[validate_checks.py]] - degree 8, connects to 1 community
- [[ProgressCallback]] - degree 7, connects to 1 community
- [[_stratified_nested_prefix_indices()]] - degree 6, connects to 1 community