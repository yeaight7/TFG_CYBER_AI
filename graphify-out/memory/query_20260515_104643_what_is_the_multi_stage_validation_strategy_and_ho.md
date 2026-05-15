---
type: "query"
date: "2026-05-15T10:46:43.865734+00:00"
question: "What is the multi-stage validation strategy and how do checks A, B, C and leave-one-out relate?"
contributor: "graphify"
source_nodes: ["check_b_shuffled_labels (anti-leakage test)", "check_c_csv_split (realistic CSV-split eval)", "validate_leave_one_csv_out main pipeline", "RLDatasetDefenderEnv gymnasium environment", "check_a_direct_eval (direct model evaluation)"]
---

# Q: What is the multi-stage validation strategy and how do checks A, B, C and leave-one-out relate?

## Answer

The project uses a 4-stage validation ladder defined in src/validate_checks.py and src/validate_leave_one_csv_out.py, all sharing RLDatasetDefenderEnv. Check A (check_a_direct_eval, line 83): direct model.predict() on test set to verify basic performance - guards against environment-dependency. Check B (check_b_shuffled_labels, line 141): trains briefly with shuffled y_train labels - if model still gets high accuracy, it has memorized the input space rather than learned. This is the anti-leakage test. Check C (check_c_csv_split, line 271): trains on a subset of CICIDS2017 CSV files and tests on held-out ones, simulating day-of-week generalization. Leave-One-Exact-CSV-Out (validate_leave_one_csv_out.py): the most rigorous - iterates all CICIDS2017 CSV files, training on all but one and testing on that one, then aggregates fold metrics via _build_aggregate_results(). The graph correctly shows all four sharing load_cicids2017_split/exact_csv_split and RLDatasetDefenderEnv as infrastructure. CICIDS2017 Training Run History cross-references all checks against the C03 best run.

## Source Nodes

- check_b_shuffled_labels (anti-leakage test)
- check_c_csv_split (realistic CSV-split eval)
- validate_leave_one_csv_out main pipeline
- RLDatasetDefenderEnv gymnasium environment
- check_a_direct_eval (direct model evaluation)