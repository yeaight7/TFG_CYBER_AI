# Thesis figure audit

Date: 2026-09-01  
Scope: current supervision draft; quantitative campaign outputs remain pending

| Figure or asset | Decision | Reason / action |
|---|---|---|
| F1 project Gantt | IMPROVE | Extended through final-campaign preparation and the supervision draft; final campaign is not shown as completed. |
| F2 system pipeline | IMPROVE | Added the unscaled cache, split/model boundary, train-only scaler, schema 3.0 artifacts, and reuse by Phase 2. |
| F3 agent--environment interaction | KEEP | Clear statement of the binary action and reward semantics; still matches the implementation. |
| F4 canonical observation | KEEP | Concise explanation of the stable 76+76 contract. |
| F5 QRDQN architecture | KEEP | Still matches the frozen `main-v1` network; caption now treats it as profile architecture rather than final-run evidence. |
| F6 CICIDS2017 day composition | KEEP | Artifact-backed dataset context that explains why day separation is meaningful. |
| F7 historical random-split balance | REMOVE FROM MANUSCRIPT | Correct but redundant with prose and not necessary in the final evidence hierarchy; source asset is retained for auditability. |
| F8 historical MAIN training curves | KEEP | Useful process evidence, now labelled historical and not used to claim final convergence. |
| F9 historical MAIN confusion matrix | KEEP | Compact historical reference; final MAIN will receive a newly generated matrix. |
| F10 historical bootstrap intervals | KEEP | Explains fixed-test precision; explicitly not training-seed variance. |
| F11 historical Check C matrix | KEEP TEMPORARILY | Documents the proxy that motivated the full day run; expected to be relegated or removed after final day evidence exists. |
| F12 historical duplicate analysis | KEEP | Material threat-to-validity evidence, clearly scoped to the historical split. |
| F13 historical QRDQN--RF comparison | REPLACE | Removed the unmatched LOO panel and the vertical “pending GPU” mark; now shows only the two historical partitions with an explicit proxy note. |
| F14 historical RF day matrix | KEEP TEMPORARILY | Helps explain the historical RF failure mode; reconsider after the matched final day figure exists. |
| F15 historical Phase 2 diagnostics | KEEP TEMPORARILY | Useful pipeline evidence with limited external validity; final fresh inference will replace it. |
| Synthetic exploratory learning curve (`f16_curva_aprendizaje_exploratoria*`) | REMOVE | Unused and explicitly non-empirical; unsuitable for a quantitative thesis figure. |
| F16 validation ladder | REPLACE | Rebuilt without historical metric callouts as a generalisation hierarchy plus orthogonal size/seed stability study. |
| Synthetic estimated learning curve (`f17_curva_aprendizaje_estimacion_no_empirica*`) | REMOVE | Unused fabricated/interpolated geometry; no place in the manuscript. |
| F17 Phase 2 pipeline | REPLACE | Rebuilt around the fresh MAIN contract, hashes, no scaler refit, diagnostics, and offline-only output. |
| F18 final campaign overview | ADD | Explains MAIN, day, ladder, seeds, holdouts, RF, aliases, and auxiliaries more clearly than prose alone. |
| F19 evidence traceability | ADD | Shows the fail-closed route from frozen specification to a thesis claim. |

## Pending final-data figures

These figures are deliberately not generated until a complete aggregate exists:

- fresh MAIN training curves and confusion matrix;
- fresh MAIN bootstrap intervals and duplicate analysis;
- full day-split QRDQN/RF comparison and final day confusion matrix;
- size-ladder learning curve;
- model-seed sensitivity distribution;
- four targeted-holdout comparison with support/prevalence context;
- matched QRDQN--Random Forest generalisation comparison;
- fresh Phase 2 metrics and distribution diagnostics.

All quantitative geometry must be rendered programmatically from validated JSON/CSV/run artifacts. Conceptual figures F1--F5 and F16--F19 encode design or relationships, not measured values.
