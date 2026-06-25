---
type: community
cohesion: 0.36
members: 8
---

# Hparam Tuning

**Cohesion:** 0.36 - loosely connected
**Members:** 8 nodes

## Members
- [[Evaluate model and return F1 score for attack class.]] - rationale - src/tune_hparams.py
- [[Optuna objective train QRDQN with suggested hparams, return F1 attack.]] - rationale - src/tune_hparams.py
- [[_evaluate_f1()]] - code - src/tune_hparams.py
- [[main()_7]] - code - src/tune_hparams.py
- [[objective()]] - code - src/tune_hparams.py
- [[parse_args()_4]] - code - src/tune_hparams.py
- [[tune_hparams.py]] - code - src/tune_hparams.py
- [[tune_hparams.py -- Optimizacion de hiperparametros con Optuna para QRDQN sobre C]] - rationale - src/tune_hparams.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Hparam_Tuning
SORT file.name ASC
```

## Connections to other communities
- 2 edges to [[_COMMUNITY_Baseline Random Forest]]
- 1 edge to [[_COMMUNITY_RL Defender Environment]]

## Top bridge nodes
- [[main()_7]] - degree 5, connects to 1 community
- [[_evaluate_f1()]] - degree 4, connects to 1 community