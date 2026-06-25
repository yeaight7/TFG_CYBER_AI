# Archived runs (pre-design probes)

These are **exploratory runs from before the experimental design was fixed** — they are **not** part of the official trunk and must **not** be presented as official results. They are kept tracked only for traceability.

The official run is `runs/cicids2017/MAIN_qrdqn_cicids2017_canonical_full_random_20260609_193655/` (profile `main-experiment`, full data, fixed test partition). See `docs/results.md` and `docs/audits/repo_cleanup_implementation_guide_2026-06-25.md` (decision 1, §4).

## Contents

- `cicids2017/` — C0x training probes (C01 smoke/full, C02 fast, C03 full, plus the C03 fast-day TensorBoard-only directory) and the three `MAIN_*fast_random_*` smoke runs from 2026-06-09. The `MAIN` prefix on those three is misleading — they are `fast` smoke runs, not the official full run.
- `phase2/` — early Phase 2 inference probes (Feb–Apr 2026). The two domain-shift exemplars still cited in the docs (`P2v2_pred_20260224_004121`, `P2v2_pred_20260408_230318`) and the official labelled run (`P2v2_pred_20260610_161231_MAIN`) intentionally remain under `runs/phase2/`.

Note: some heavy/duplicated artifacts here (TensorBoard `events.out.tfevents.*`, per-run `model.zip` files that duplicate `models/`) are slated for git de-tracking in a later cleanup phase; the archival move itself keeps them tracked.
