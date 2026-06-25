---
type: community
cohesion: 0.15
members: 20
---

# Testing Splits

**Cohesion:** 0.15 - loosely connected
**Members:** 20 nodes

## Members
- [[Deterministic, stratified, nested train subsample indices.      Per-class se]] - rationale - src/load_cicids2017.py
- [[SHA-256 content hash of an ndarray, prefixed with dtype and shape so that     a]] - rationale - src/load_cicids2017.py
- [[_sha256_of_array()]] - code - src/load_cicids2017.py
- [[_split()]] - code - tests/test_load_cicids2017.py
- [[_stratified_nested_prefix_indices()]] - code - src/load_cicids2017.py
- [[_synthetic_df()]] - code - tests/test_load_cicids2017.py
- [[fail()]] - code - scripts/verify_fixed_test_split.py
- [[main()_4]] - code - scripts/verify_fixed_test_split.py
- [[ok()]] - code - scripts/verify_fixed_test_split.py
- [[parse_args()_2]] - code - scripts/verify_fixed_test_split.py
- [[patched_loader()]] - code - tests/test_load_cicids2017.py
- [[test_load_cicids2017.py]] - code - tests/test_load_cicids2017.py
- [[test_metadata_new_keys()]] - code - tests/test_load_cicids2017.py
- [[test_nested_prefix_indices_deterministic()]] - code - tests/test_load_cicids2017.py
- [[test_nested_prefix_indices_nested_and_stratified()]] - code - tests/test_load_cicids2017.py
- [[test_scale_true_refits_on_subsample()]] - code - tests/test_load_cicids2017.py
- [[test_sha256_of_array_stable()]] - code - tests/test_load_cicids2017.py
- [[test_train_max_rows_guards()]] - code - tests/test_load_cicids2017.py
- [[test_train_max_rows_keeps_test_set_identical()]] - code - tests/test_load_cicids2017.py
- [[verify_fixed_test_split.py]] - code - scripts/verify_fixed_test_split.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Testing_Splits
SORT file.name ASC
```

## Connections to other communities
- 7 edges to [[_COMMUNITY_Baseline Random Forest]]

## Top bridge nodes
- [[test_load_cicids2017.py]] - degree 11, connects to 1 community
- [[main()_4]] - degree 7, connects to 1 community
- [[_stratified_nested_prefix_indices()]] - degree 6, connects to 1 community
- [[_split()]] - degree 6, connects to 1 community
- [[_sha256_of_array()]] - degree 5, connects to 1 community