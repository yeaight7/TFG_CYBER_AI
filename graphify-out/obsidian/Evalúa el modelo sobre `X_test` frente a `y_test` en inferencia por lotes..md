---
source_file: "src\validate_leave_one_csv_out.py"
type: "rationale"
community: "Leave-One-CSV Validation"
location: "L156"
tags:
  - graphify/rationale
  - graphify/INFERRED
  - community/Leave-One-CSV_Validation
---

# Evalúa el modelo sobre `X_test` frente a `y_test` en inferencia por lotes.

## Connections
- [[CICIDSLoadConfig]] - `uses` [INFERRED]
- [[RLDatasetDefenderEnv]] - `uses` [INFERRED]
- [[evaluate_model_direct()]] - `rationale_for` [EXTRACTED]

#graphify/rationale #graphify/INFERRED #community/Leave-One-CSV_Validation