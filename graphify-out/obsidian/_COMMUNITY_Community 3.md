---
type: community
members: 20
---

# Community 3

**Members:** 20 nodes

## Members
- [[152-D Observation Vector]] - document - .github/AGENT_CONTEXT.md
- [[76 Canonical Flow Features]] - document - .github/AGENT_CONTEXT.md
- [[Canonical Flow Schema]] - document - .github/AGENT_CONTEXT.md
- [[CanonicalResult]] - code - src/canonical_schema.py
- [[Dataset Adapter Contract]] - document - .github/AGENT_CONTEXT.md
- [[Devuelve la lista completa de nombres features + máscara de missingness.]] - rationale - src/canonical_schema.py
- [[Devuelve la lista de nombres de features canónicas (sin máscara).]] - rationale - src/canonical_schema.py
- [[Mapea un DataFrame al esquema canónico de features.      Parameters     -----]] - rationale - src/canonical_schema.py
- [[Missingness Mask]] - document - .github/AGENT_CONTEXT.md
- [[Resultado de mapear un DataFrame al esquema canónico.]] - rationale - src/canonical_schema.py
- [[batched_predict()]] - code - scripts/predict_real_traffic.py
- [[canonical_schema.py]] - code - src/canonical_schema.py
- [[canonical_schema.py — Definición formal del esquema canónico de features (FEATUR]] - rationale - src/canonical_schema.py
- [[get_canonical_feature_names()]] - code - src/canonical_schema.py
- [[get_observation_feature_names()]] - code - src/canonical_schema.py
- [[load_model()]] - code - scripts/predict_real_traffic.py
- [[main()_1]] - code - scripts/predict_real_traffic.py
- [[map_to_canonical()]] - code - src/canonical_schema.py
- [[maybe_convert_time_units()]] - code - scripts/predict_real_traffic.py
- [[predict_real_traffic.py]] - code - scripts/predict_real_traffic.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Community_3
SORT file.name ASC
```

## Connections to other communities
- 3 edges to [[_COMMUNITY_Community 0]]
- 3 edges to [[_COMMUNITY_Community 6]]
- 1 edge to [[_COMMUNITY_Community 1]]
- 1 edge to [[_COMMUNITY_Community 2]]

## Top bridge nodes
- [[map_to_canonical()]] - degree 8, connects to 3 communities
- [[Canonical Flow Schema]] - degree 8, connects to 3 communities
- [[get_observation_feature_names()]] - degree 4, connects to 1 community
- [[Dataset Adapter Contract]] - degree 2, connects to 1 community