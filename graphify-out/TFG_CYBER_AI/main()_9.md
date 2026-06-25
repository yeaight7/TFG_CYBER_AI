---
source_file: "src/validate_leave_one_csv_out.py"
type: "code"
community: "Leave-One-Out Eval"
location: "L311"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/Leave-One-Out_Eval
---

# main()

## Connections
- [[CICIDSLoadConfig]] - `calls` [INFERRED]
- [[_build_aggregate_results()]] - `calls` [EXTRACTED]
- [[_resolve_holdout_csvs()]] - `calls` [EXTRACTED]
- [[evaluate_model_direct()]] - `calls` [EXTRACTED]
- [[list_cicids2017_csv_files()]] - `calls` [INFERRED]
- [[load_cicids2017_exact_csv_split()]] - `calls` [INFERRED]
- [[make_env_fn()_1]] - `calls` [EXTRACTED]
- [[parse_args()_6]] - `calls` [EXTRACTED]
- [[validate_leave_one_csv_out.py]] - `contains` [EXTRACTED]

#graphify/code #graphify/EXTRACTED #community/Leave-One-Out_Eval