---
type: community
members: 35
---

# RF Baseline Module

**Members:** 35 nodes

## Members
- [[CICIDSLoadConfig]] - code - src/load_cicids2017.py
- [[Carga CICIDS2017 desde directorio local y lo adapta al esquema canónico.]] - rationale - src/load_cicids2017.py
- [[Carga CICIDS2017 separando por archivos CSV (train vs test).      En lugar de]] - rationale - src/load_cicids2017.py
- [[Carga CICIDS2017 separando por nombres exactos de archivo CSV.      A diferenc]] - rationale - src/load_cicids2017.py
- [[Carga un CSV individual con límite opcional de filas por archivo.]] - rationale - src/load_cicids2017.py
- [[Carga una lista de CSVs, aplica preprocesado y devuelve X, y y features.]] - rationale - src/load_cicids2017.py
- [[Entrena un RandomForestClassifier sobre el dataset y devuelve el modelo.]] - rationale - src/baseline_random_forest.py
- [[Evalúa el Random Forest y muestra matriz de confusión + informe de clasificación]] - rationale - src/baseline_random_forest.py
- [[Limpia CICIDS2017 y devuelve X, y y nombres de features.]] - rationale - src/load_cicids2017.py
- [[Lista únicamente los 8 CSVs oficiales de CICIDS2017 en orden determinista.]] - rationale - src/load_cicids2017.py
- [[Resuelve nombres exactos de CSV a rutas reales, preservando el orden de entrada.]] - rationale - src/load_cicids2017.py
- [[Unified CICIDS2017 loader with split-mode and preset support.      Parameters]] - rationale - src/load_cicids2017.py
- [[_clean_rows()]] - code - src/load_cicids2017.py
- [[_coerce_numeric_features()]] - code - src/load_cicids2017.py
- [[_drop_identifier_like_columns()]] - code - src/load_cicids2017.py
- [[_find_label_column()]] - code - src/load_cicids2017.py
- [[_list_csv_files()]] - code - src/load_cicids2017.py
- [[_load_all_csvs()]] - code - src/load_cicids2017.py
- [[_load_and_process_csv_paths()]] - code - src/load_cicids2017.py
- [[_load_csv_with_row_limit()]] - code - src/load_cicids2017.py
- [[_normalize_columns()]] - code - src/load_cicids2017.py
- [[_prepare_cicids_features()]] - code - src/load_cicids2017.py
- [[_resolve_exact_csv_names()]] - code - src/load_cicids2017.py
- [[baseline_random_forest.py]] - code - src/baseline_random_forest.py
- [[evaluate_random_forest()]] - code - src/baseline_random_forest.py
- [[list_cicids2017_csv_files()]] - code - src/load_cicids2017.py
- [[load_cicids2017.py]] - code - src/load_cicids2017.py
- [[load_cicids2017_binary()]] - code - src/load_cicids2017.py
- [[load_cicids2017_csv_split()]] - code - src/load_cicids2017.py
- [[load_cicids2017_exact_csv_split()]] - code - src/load_cicids2017.py
- [[load_cicids2017_split()]] - code - src/load_cicids2017.py
- [[main()_3]] - code - src/baseline_random_forest.py
- [[test_load_cicids2017.py]] - code - tests/test_load_cicids2017.py
- [[test_prepare_cicids_features_binary_labels()]] - code - tests/test_load_cicids2017.py
- [[train_random_forest()]] - code - src/baseline_random_forest.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/RF_Baseline_Module
SORT file.name ASC
```

## Connections to other communities
- 3 edges to [[_COMMUNITY_Leave-One-Out Validation]]
- 2 edges to [[_COMMUNITY_Hyperparameter Tuning]]
- 2 edges to [[_COMMUNITY_Validation Checks Module]]
- 1 edge to [[_COMMUNITY_Deprecated Inference v1]]
- 1 edge to [[_COMMUNITY_RL Environment (Code)]]

## Top bridge nodes
- [[CICIDSLoadConfig]] - degree 9, connects to 2 communities
- [[load_cicids2017_split()]] - degree 8, connects to 2 communities
- [[_prepare_cicids_features()]] - degree 10, connects to 1 community
- [[load_cicids2017_binary()]] - degree 9, connects to 1 community
- [[load_cicids2017_exact_csv_split()]] - degree 8, connects to 1 community