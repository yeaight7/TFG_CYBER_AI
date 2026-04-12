---
type: community
members: 27
---

# CICIDS2017 Loader

**Members:** 27 nodes

## Members
- [[CICIDSLoadConfig]] - code - src\load_cicids2017.py
- [[Carga CICIDS2017 desde directorio local y lo adapta al esquema canónico.]] - rationale - src\load_cicids2017.py
- [[Carga CICIDS2017 separando por archivos CSV (train vs test).      En lugar de]] - rationale - src\load_cicids2017.py
- [[Carga CICIDS2017 separando por nombres exactos de archivo CSV.      A diferenc]] - rationale - src\load_cicids2017.py
- [[Carga un CSV individual con límite opcional de filas por archivo.]] - rationale - src\load_cicids2017.py
- [[Carga una lista de CSVs, aplica preprocesado y devuelve X, y y features.]] - rationale - src\load_cicids2017.py
- [[Limpia CICIDS2017 y devuelve X, y y nombres de features.]] - rationale - src\load_cicids2017.py
- [[Lista únicamente los 8 CSVs oficiales de CICIDS2017 en orden determinista.]] - rationale - src\load_cicids2017.py
- [[Resuelve nombres exactos de CSV a rutas reales, preservando el orden de entrada.]] - rationale - src\load_cicids2017.py
- [[Unified CICIDS2017 loader with split-mode and preset support.      Parameters]] - rationale - src\load_cicids2017.py
- [[_clean_rows()]] - code - src\load_cicids2017.py
- [[_coerce_numeric_features()]] - code - src\load_cicids2017.py
- [[_drop_identifier_like_columns()]] - code - src\load_cicids2017.py
- [[_find_label_column()]] - code - src\load_cicids2017.py
- [[_list_csv_files()]] - code - src\load_cicids2017.py
- [[_load_all_csvs()]] - code - src\load_cicids2017.py
- [[_load_and_process_csv_paths()]] - code - src\load_cicids2017.py
- [[_load_csv_with_row_limit()]] - code - src\load_cicids2017.py
- [[_normalize_columns()]] - code - src\load_cicids2017.py
- [[_prepare_cicids_features()]] - code - src\load_cicids2017.py
- [[_resolve_exact_csv_names()]] - code - src\load_cicids2017.py
- [[list_cicids2017_csv_files()]] - code - src\load_cicids2017.py
- [[load_cicids2017.py]] - code - src\load_cicids2017.py
- [[load_cicids2017_binary()]] - code - src\load_cicids2017.py
- [[load_cicids2017_csv_split()]] - code - src\load_cicids2017.py
- [[load_cicids2017_exact_csv_split()]] - code - src\load_cicids2017.py
- [[load_cicids2017_split()]] - code - src\load_cicids2017.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/CICIDS2017_Loader
SORT file.name ASC
```

## Connections to other communities
- 9 edges to [[_COMMUNITY_Environment and Tuning]]
- 3 edges to [[_COMMUNITY_QRDQN Training]]
- 3 edges to [[_COMMUNITY_Leave-One-CSV Validation]]
- 2 edges to [[_COMMUNITY_Random Forest Baseline]]
- 2 edges to [[_COMMUNITY_Canonical Schema Guardrails]]
- 1 edge to [[_COMMUNITY_Project Docs and Phase Plan]]

## Top bridge nodes
- [[CICIDSLoadConfig]] - degree 22, connects to 4 communities
- [[load_cicids2017.py]] - degree 20, connects to 2 communities