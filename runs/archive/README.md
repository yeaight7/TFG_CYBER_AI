# Archived runs (pre-design probes)

These are **exploratory runs from before the experimental design was fixed** — they are **not** part of the official trunk and must **not** be presented as official results. They are kept tracked only for traceability.

The official run is `runs/cicids2017/MAIN_qrdqn_cicids2017_canonical_full_random_20260609_193655/` (profile `main-experiment`, full data, fixed test partition). See `docs/results.md` and `docs/audits/repo_cleanup_implementation_guide_2026-06-25.md` (decision 1, §4).

## Contents

- `cicids2017/` — C0x training probes (C01 smoke/full, C02 fast, C03 full) and the three `MAIN_*fast_random_*` smoke runs from 2026-06-09. The `MAIN` prefix on those three is misleading — they are `fast` smoke runs, not the official full run.
- `phase2/` — early Phase 2 inference probes (Feb–Apr 2026). The two domain-shift exemplars still cited in the docs (`P2v2_pred_20260224_004121`, `P2v2_pred_20260408_230318`) and the official labelled run (`P2v2_pred_20260610_161231_MAIN`) intentionally remain under `runs/phase2/`.

Note: the heavy/duplicated artifacts that once lived here (TensorBoard `events.out.tfevents.*`, per-run `model.zip` files duplicating `models/`) were de-tracked in earlier cleanup phases (gitignored via `runs/**/events.out.tfevents.*` and `runs/**/model.zip`), and the `models/archive/` + `pcaps/archive/` binaries were de-tracked in 2026-07. Everything remains on local disk and in git history (decision D-6, no rewrite).
