# Archived models (pre-design probes)

Exploratory model checkpoints from **before the experimental design was fixed** — **not official**, kept tracked only for traceability.

The official model is `models/MAIN_qrdqn_cicids2017_canonical_full_random_20260609_193655.zip`. See `docs/results.md` and `docs/audits/repo_cleanup_implementation_guide_2026-06-25.md` (decision 1, §4).

## Contents

- `C01_*`, `C02_*`, `C03_*` — CICIDS2017 QRDQN pre-design probes.
- `A01__arch256x256_*`, `A02_dqn_arch512x256_*` — early architecture experiments.
- `rl_defender_dqn.zip` — early DQN defender prototype.
- `MAIN_*fast_random_*.zip` — `fast` smoke runs (misleading `MAIN` prefix; not the official full run).

## Tracking status (2026-07)

The `.zip` binaries in this directory are no longer git-tracked (`git rm
--cached`, 2026-07): they remain on this machine's disk and in git history
(decision D-6, no rewrite), but do not ship in fresh clones. This README
stays tracked as the record of what the archive contains.
