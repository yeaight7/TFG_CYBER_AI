---
type: community
members: 22
---

# Deprecated Inference v1

**Members:** 22 nodes

## Members
- [[CanonicalResult]] - code - src/canonical_schema.py
- [[Carga NSL-KDD localmente, lo preprocesa y devuelve          X_train, y_train,]] - rationale - src/load_nsl_kdd.py
- [[Devuelve la lista completa de nombres features + máscara de missingness.]] - rationale - src/canonical_schema.py
- [[Devuelve la lista de nombres de features canónicas (sin máscara).]] - rationale - src/canonical_schema.py
- [[Mapea un DataFrame al esquema canónico de features.      Parameters     -----]] - rationale - src/canonical_schema.py
- [[Resultado de mapear un DataFrame al esquema canónico.]] - rationale - src/canonical_schema.py
- [[_ensure_dataset_local_dir()]] - code - src/load_nsl_kdd.py
- [[batched_predict()]] - code - scripts/deprecated_predict_real_traffic.py
- [[canonical_schema.py]] - code - src/canonical_schema.py
- [[canonical_schema.py — Definición formal del esquema canónico de features (FEATUR]] - rationale - src/canonical_schema.py
- [[deprecated_predict_real_traffic.py]] - code - scripts/deprecated_predict_real_traffic.py
- [[get_canonical_feature_names()]] - code - src/canonical_schema.py
- [[get_observation_feature_names()]] - code - src/canonical_schema.py
- [[load_model()]] - code - scripts/deprecated_predict_real_traffic.py
- [[load_nsl_kdd.py]] - code - src/load_nsl_kdd.py
- [[load_nsl_kdd_binary()]] - code - src/load_nsl_kdd.py
- [[main()]] - code - scripts/deprecated_predict_real_traffic.py
- [[map_to_canonical()]] - code - src/canonical_schema.py
- [[maybe_convert_time_units()]] - code - scripts/deprecated_predict_real_traffic.py
- [[test_canonical_features_length()]] - code - tests/test_canonical_schema.py
- [[test_canonical_schema.py]] - code - tests/test_canonical_schema.py
- [[test_map_to_canonical_mask_logic()]] - code - tests/test_canonical_schema.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Deprecated_Inference_v1
SORT file.name ASC
```

## Connections to other communities
- 1 edge to [[_COMMUNITY_Scaling & Clipping Utilities]]
- 1 edge to [[_COMMUNITY_RF Baseline Module]]

## Top bridge nodes
- [[map_to_canonical()]] - degree 8, connects to 2 communities