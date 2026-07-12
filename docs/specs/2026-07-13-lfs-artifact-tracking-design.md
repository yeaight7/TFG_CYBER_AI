# LFS and Experimental Artifact Tracking Design

**Status:** Approved design
**Date:** 2026-07-13

## Objective

Make fresh clones clean and make reproducibility-critical experiment artifacts
committable without rewriting Git history. Large opaque artifacts use Git LFS;
small human-readable evidence stays in normal Git; future per-row prediction CSV
exports and downloaded bundles remain outside version control.

## Current Problem

The repository declares all ZIP and joblib files as LFS-managed, but eight
currently tracked files are raw Git blobs. Git LFS clean filters therefore make
fresh WSL clones report those files as modified even when no user edit occurred.

Separately, `.gitignore` excludes run-local `model.zip`, TensorBoard event files,
and checkpoint directories. Those exclusions prevent complete future campaign
evidence from being staged normally.

## Decisions

### Forward-only normalization

Normalize the eight raw ZIP/joblib files in the current tree into LFS pointers in
a new commit. Preserve their materialized bytes and SHA-256 values. Do not use
`git lfs migrate import`, rewrite history, delete historical blobs, or change old
commit IDs.

### Artifact policy

| Artifact family | Policy | Reason |
|---|---|---|
| Final model ZIPs | Git LFS | Required evidence; binary and potentially large |
| Retained checkpoint ZIPs | Git LFS | Bounded recovery/diagnostic evidence |
| TensorBoard event files | Git LFS | Required binary training trace |
| Joblib models and scalers | Git LFS | Binary model/preprocessing evidence |
| Official CICIDS2017 CSVs | Git LFS | Large source data already managed through LFS |
| Existing committed prediction CSVs | Keep tracked through LFS | Preserve historical evidence unchanged |
| Existing/future TensorBoard scalar CSVs | Git LFS | Preserve current representation and generated traces |
| PCAPs and already tracked PCAP-derived CSVs | Git LFS | Preserve existing binary/large-input policy |
| Config, metrics, environment, timing, monitoring, manifests, checksums, logs | Normal Git | Small, human-readable provenance |
| `system_metrics.csv`, feature importances, aggregate/summary CSVs | Normal Git | Small, reviewable evidence |
| `predictions.npz`, percentiles, feature names | Normal Git | Required compact evidence |
| Future per-row `runs/**/predictions*.csv` | Ignored | Large rebuildable/exportable representation |
| Sensitive prediction CSVs | Ignored | Existing privacy boundary |
| Downloaded `.tar.gz` bundles | Ignored | Durable transfer artifact, not repository content |
| Canonical cache arrays | Ignored | Rebuildable from source data |
| Replay buffers | Ignored/disabled | Not part of the approved campaign contract |

An ignore rule does not untrack an already committed file. Existing prediction
CSVs therefore remain in the index while new matching exports stay untracked.

### CSV attribute scope

Replace the global `*.csv` LFS rule with path-scoped rules that continue to cover
all currently LFS-managed CSVs:

- official CICIDS2017 dataset CSVs;
- PCAP-derived CSV evidence;
- existing committed prediction CSVs;
- TensorBoard scalar CSVs.

This keeps `system_metrics.csv`, RF feature importance CSVs, and campaign
aggregate CSVs in normal Git.

### Run output location

Git can only stage artifacts written beneath the repository. GPU-host campaign
commands intended for later evidence commits must place `--artifact-root` under a
tracked repository path such as `runs/final_campaign/`. Snapshot and final bundle
destinations remain separate durable storage locations; Git does not replace the
verified snapshot/export workflow.

## Files and Index Entries

Expected source changes:

- `.gitattributes`;
- `.gitignore`;
- `docs/reproducibility.md`;
- a focused artifact-policy regression test.

Expected index-only normalization:

- the two current raw model files under `models/`;
- the six current raw scaler joblib files under `runs/`.

The working-tree bytes of those eight files must not change.

## Validation

1. Record SHA-256 values for the eight inconsistent files before normalization.
2. Apply attributes and normalize only the named files.
3. Recompute SHA-256 values and require exact equality.
4. Verify representative paths with `git check-attr` and `git check-ignore`:
   final models, checkpoints, TensorBoard events, logs, system metrics, prediction
   CSVs, bundles, dataset CSVs, and cache paths.
5. Run `git lfs status`, `git lfs ls-files`, and `git lfs fsck`.
6. Run the focused artifact-policy test, complete unit suite, Ruff, and
   `git diff --check`.
7. Verify a fresh checkout/clone is clean and materializes the normalized files
   through LFS.

## Failure Handling

- Abort if any normalized working-file hash changes.
- Abort if existing committed prediction CSVs disappear from the index.
- Abort if important run evidence remains ignored.
- Abort if a representative generated prediction CSV becomes stageable without
  force.
- Do not push if any required LFS object is missing or corrupt.

## Non-goals

- No history rewrite or force-push.
- No deletion or semantic modification of historical artifacts.
- No campaign execution or generated result claims.
- No change to scientific profiles, seeds, preprocessing, or model behavior.
- No automatic cloud upload implementation.
