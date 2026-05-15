---
type: "query"
date: "2026-05-15T10:46:31.363770+00:00"
question: "How does the canonical schema connect to the RL environment and training pipeline?"
contributor: "graphify"
source_nodes: ["map_to_canonical function", "CanonicalResult dataclass", "RLDatasetDefenderEnv gymnasium environment", "load_cicids2017__prepare_cicids_features", "canonical_schema_missingness_mask"]
---

# Q: How does the canonical schema connect to the RL environment and training pipeline?

## Answer

The connection is a direct 3-hop chain: (1) map_to_canonical() (src/canonical_schema.py:277) accepts a raw DataFrame from either CICIDS2017 or NSL-KDD, applies the dataset-specific mapping dict (CICIDS2017_TO_CANON or NSL_KDD_TO_CANON), and returns a CanonicalResult with 76 features + a 76-dim missingness mask. (2) _prepare_cicids_features() (src/load_cicids2017.py) calls map_to_canonical() and feeds cleaned arrays to load_cicids2017_binary/split. (3) make_env_fn() in train_rl_defender.py constructs RLDatasetDefenderEnv with that data. The env's _get_observation() concatenates the 76 features with the 76-dim mask to produce a 152-dim observation vector consumed by the QRDQN agent. The missingness mask is a deliberate design: it lets the RL agent learn that missing features carry information (e.g. a field absent only in attack flows). The FEATURES_CANON list enforces consistent feature ordering across both datasets.

## Source Nodes

- map_to_canonical function
- CanonicalResult dataclass
- RLDatasetDefenderEnv gymnasium environment
- load_cicids2017__prepare_cicids_features
- canonical_schema_missingness_mask