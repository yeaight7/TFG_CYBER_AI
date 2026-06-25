---
type: community
cohesion: 0.15
members: 17
---

# RL Defender Environment

**Cohesion:** 0.15 - loosely connected
**Members:** 17 nodes

## Members
- [[.__init__()]] - code - src/rl_defender_env.py
- [[._compute_reward()]] - code - src/rl_defender_env.py
- [[._get_observation()]] - code - src/rl_defender_env.py
- [[.close()]] - code - src/rl_defender_env.py
- [[.render()]] - code - src/rl_defender_env.py
- [[.reset()]] - code - src/rl_defender_env.py
- [[.step()]] - code - src/rl_defender_env.py
- [[Calcula la recompensa en función de la etiqueta real, la acción         y la co]] - rationale - src/rl_defender_env.py
- [[Entorno RL para un defensor que decide PERMITBLOCK sobre muestras etiquetadas.]] - rationale - src/rl_defender_env.py
- [[RLDatasetDefenderEnv]] - code - src/rl_defender_env.py
- [[rl_defender_env.py]] - code - src/rl_defender_env.py
- [[test_env_initialization()]] - code - tests/test_rl_defender_env.py
- [[test_env_step_and_reset()]] - code - tests/test_rl_defender_env.py
- [[test_reward_config.py]] - code - tests/test_reward_config.py
- [[test_reward_logic_tp_fp_tn_fn()]] - code - tests/test_reward_config.py
- [[test_rl_defender_env.py]] - code - tests/test_rl_defender_env.py
- [[test_unknown_label_reward()]] - code - tests/test_reward_config.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/RL_Defender_Environment
SORT file.name ASC
```

## Connections to other communities
- 1 edge to [[_COMMUNITY_Validation Checks]]
- 1 edge to [[_COMMUNITY_Hparam Tuning]]

## Top bridge nodes
- [[RLDatasetDefenderEnv]] - degree 15, connects to 2 communities