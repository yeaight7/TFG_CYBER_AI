---
type: community
cohesion: 0.09
members: 38
---

# Project Overview

**Cohesion:** 0.09 - loosely connected
**Members:** 38 nodes

## Members
- [[AGENTS]] - document - AGENTS.md
- [[AGENT_CONTEXT]] - document - docs/AGENT_CONTEXT.md
- [[Anti-Leakage Policy (drop IPs, timestamps, Flow IDs, ports)]] - rationale - docs/Personal Research/deep-defense-research/02-datos-esquema-canonico-y-preprocesado.md
- [[Binary Action Space PERMIT(0)  BLOCK(1)]] - rationale - docs/Personal Research/deep-defense-research/01-fundamentos-y-objetivo.md
- [[CICIDS2017 Dataset]] - paper - GEMINI.md
- [[CICIDS2017 Dataset (8 CSVs, 5-day capture)]] - document - docs/Personal Research/deep-defense-research/02-datos-esquema-canonico-y-preprocesado.md
- [[DEFENSA_TFG_PROGRESO]] - document - docs/DEFENSA_TFG_PROGRESO.md
- [[DEFENSA_TFG_SCRIPT]] - document - docs/DEFENSA_TFG_SCRIPT.md
- [[Deep Defense Datos, Esquema Canonico y Preprocesado]] - document - docs/Personal Research/deep-defense-research/02-datos-esquema-canonico-y-preprocesado.md
- [[Deep Defense Fundamentos y Objetivo]] - document - docs/Personal Research/deep-defense-research/01-fundamentos-y-objetivo.md
- [[Deep Defense Glosario y Preguntas de Tribunal]] - document - docs/Personal Research/deep-defense-research/06-glosario-y-preguntas-tribunal.md
- [[Deep Defense Phase 2 - Laboratorio, Inferencia y Riesgos]] - document - docs/Personal Research/deep-defense-research/05-phase2-laboratorio-inferencia-y-riesgos.md
- [[Domain Shift]] - paper - docs/AGENT_CONTEXT.md
- [[FEATURES_CANON (Canonical Schema)]] - paper - AGENTS.md
- [[GEMINI]] - document - GEMINI.md
- [[Leave-one-exact-CSV-out Validation]] - paper - src/validate_leave_one_csv_out.py
- [[Missingness Mask]] - paper - AGENTS.md
- [[NSL-KDD Dataset]] - paper - GEMINI.md
- [[Phase 2 Offline Inference on Lab Traffic]] - rationale - docs/Personal Research/deep-defense-research/05-phase2-laboratorio-inferencia-y-riesgos.md
- [[Private Lab]] - paper - docs/gcp_lab.md
- [[QRDQN RL Algorithm]] - paper - src/train_rl_defender.py
- [[README]] - document - README.md
- [[REWARD_CONFIG]] - paper - src/train_rl_defender.py
- [[RLDatasetDefenderEnv_1]] - paper - src/train_rl_defender.py
- [[Validation Check A (Direct Eval)]] - paper - src/validate_checks.py
- [[Validation Check B (Shuffled Labels)]] - paper - src/validate_checks.py
- [[Validation Check C (CSV Split)]] - paper - src/validate_checks.py
- [[baseline_random_forest.py_1]] - code - src/baseline_random_forest.py
- [[gcp_lab]] - document - docs/gcp_lab.md
- [[load_cicids2017.py_1]] - code - src/load_cicids2017.py
- [[phase2_plan]] - document - docs/phase2_plan.md
- [[predict_real_traffic_v2.py_1]] - paper - docs/AGENT_CONTEXT.md
- [[test_load_cicids2017.py_1]] - code - tests/test_load_cicids2017.py
- [[test_train_rl_defender_config.py_1]] - code - tests/test_train_rl_defender_config.py
- [[train_rl_defender.py_1]] - code - src/train_rl_defender.py
- [[validate_checks.py_1]] - code - src/validate_checks.py
- [[validate_leave_one_csv_out.py_1]] - code - src/validate_leave_one_csv_out.py
- [[verify_fixed_test_split.py_1]] - code - scripts/verify_fixed_test_split.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Project_Overview
SORT file.name ASC
```

## Connections to other communities
- 2 edges to [[_COMMUNITY_RL Defender Training]]
- 1 edge to [[_COMMUNITY_Canonical Schema Setup]]
- 1 edge to [[_COMMUNITY_CICIDS2017 Preprocessing]]
- 1 edge to [[_COMMUNITY_Hyperparam Tuning]]
- 1 edge to [[_COMMUNITY_Test Split Validation]]
- 1 edge to [[_COMMUNITY_Leave One CSV Out]]
- 1 edge to [[_COMMUNITY_Inference Diagnostics]]

## Top bridge nodes
- [[load_cicids2017.py_1]] - degree 12, connects to 6 communities
- [[train_rl_defender.py_1]] - degree 8, connects to 1 community
- [[Deep Defense Datos, Esquema Canonico y Preprocesado]] - degree 4, connects to 1 community