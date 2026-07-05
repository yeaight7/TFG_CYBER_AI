# runs/ — Run Artifact Index

Every training/evaluation run persists `config.json` + `metrics.json` (or
`validation_results.json`) under `runs/<category>/<RUN_ID>/`. Authoritative
result claims live in [../docs/results.md](../docs/results.md) and must cite
an artifact here.

| Category | Contents | Status |
|----------|----------|--------|
| `cicids2017/` | `MAIN_qrdqn_cicids2017_canonical_full_random_20260609_193655/` — **the official run** (see `docs/results.md`); `baseline_random_forest_comparison/` — RF baseline sweeps (random/day/leave-one-out splits; its `results_rf.txt` is a **superseded unbalanced prototype**, kept for history); `test_partition_reference_seed42.json` — fixed test-partition manifest used by `scripts/verify_fixed_test_split.py`. | Official trunk |
| `validation/` | Checks A/B/C artifacts cited in `docs/results.md`: `VAL_checks_A_20260212_235443`, `VAL_checks_B_20260212_235736`, `VAL_checks_C_20260213_004847`; plus `bootstrap_ci_seed42.json` and `duplicate_analysis_seed42.json`. `VAL_checks_A_20260213_085434` and `VAL_checks_B_20260213_085502` are **duplicate re-runs cited nowhere** — kept for traceability only. | Official + 2 orphans |
| `phase2/` | Offline lab-traffic inference: `P2v2_pred_20260610_161231_MAIN` (official labelled run) and two earlier domain-shift exemplars still cited in the docs (`P2v2_pred_20260224_004121`, `P2v2_pred_20260408_230318`). | Official |
| `optuna/` | `study_20260212_222134.json` — exploratory pre-design hyperparameter study; **not used by the MAIN run**, cited nowhere. Kept for traceability. | Orphan (historical) |
| `archive/` | Pre-design probes (C0x, `MAIN_*fast*` smoke runs, early Phase-2 probes). Not official results — see [archive/README.md](archive/README.md). | Historical |
