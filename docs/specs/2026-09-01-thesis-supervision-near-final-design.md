# Thesis supervision near-final design

Date: 2026-09-01  
Status: approved for implementation  
Scope: manuscript, thesis figures, figure tooling, bibliography, and thesis-supporting documentation only

## Goal

Produce a supervision-ready manuscript that is structurally close to the final TFG while remaining explicit that the approved final experimental campaign has not yet produced evidence. The pass must synchronize the scientific method with the current repository without changing the experimental design, implementation, reward semantics, feature contract, or frozen `main-v1` profile.

## Evidence architecture

The manuscript will separate every empirical statement into three layers:

1. **Stable method and design.** Final-quality prose for the problem, canonical 76+76 observation, anti-leakage controls, cost-sensitive PERMIT/BLOCK task, reward, contextual-bandit interpretation, QRDQN, train-only preprocessing, reproducibility, ethical scope, and offline-only limitations.
2. **Historical pre-campaign evidence.** Existing MAIN, checks, Random Forest, bootstrap, duplicate, and Phase 2 artifacts remain usable only when labelled as historical or exploratory evidence that the final campaign is designed to supersede.
3. **Pending final-campaign evidence.** Fresh MAIN, full day split, size ladder, seed sensitivity, four targeted holdouts, matched RF runs, direct validation, bootstrap, duplicate analysis, shuffled-label control, and fresh Phase 2 inference remain unresolved until their committed artifacts exist.

No numeric result will cross from the third layer into the first two by inference, estimation, interpolation, or illustrative fake data.

## Manuscript structure

- The abstract will foreground the methodological contribution and replace its current result-heavy ending with a compact evidence-status statement that can later be updated from final artifacts.
- Methods will describe the approved campaign exactly: separate split/model seeds, frozen `main-v1`, unscaled canonical cache, run schema, provider-neutral GPU preflight, fixed 1M seed study, full 3M day split, size ladder, four targeted holdouts, RF scope, and artifact export policy.
- Results will use a final insertion order: fresh MAIN and controls; full day split; size ladder; seed sensitivity; targeted holdouts; matched RF comparisons; fresh Phase 2 inference. Historical results will be retained in a clearly bounded pre-campaign section.
- Discussion will be thematic rather than a metric recap: in-distribution versus generalisation, duplicates, cost-sensitive errors, QRDQN versus RF, contextual-bandit implications, size/seed/holdout sensitivity, Phase 2 shift, and threats to validity.
- Conclusions will answer what was built and learned now, distinguish supported from conditional conclusions, and state precisely what the final campaign must still decide.
- Appendices will retain scientifically useful reproduction detail while removing provider-specific operational narration.

## Pending-evidence presentation

Large visible “figure pending” and “table pending” boxes will be removed from the compiled thesis. Pending evidence will instead be represented by short, consistently styled notes or by prose that names the exact artifact contract and later insertion point. Empty quantitative axes and fake table values are prohibited.

## Figure strategy

- Keep conceptual figures that materially explain the system, observation, agent/environment relationship, and QRDQN architecture.
- Improve campaign, validation, preprocessing, Phase 2, and traceability diagrams so they match the current repository.
- Retain historical quantitative plots only where they support a bounded pre-campaign argument, with their status made visually and textually explicit.
- Remove synthetic non-empirical learning-curve assets.
- Add fail-closed rendering support for final quantitative figures. The renderer must consume validated artifacts, preserve exact data geometry, emit vector and preview formats, and refuse incomplete campaigns.
- Record a per-figure KEEP/IMPROVE/REPLACE/REMOVE/PENDING FINAL DATA audit.

## Bibliography policy

Correct only metadata verified against reliable publication records. Remove unverifiable draft citations from maintained prose instead of inventing authors, venue data, or identifiers. Report unresolved items explicitly.

## Verification

The final pass will:

- build the complete thesis from a clean LaTeX state;
- inspect undefined references/citations and layout warnings;
- verify contents, figures, tables, and page count;
- render representative pages from every chapter and every changed or added figure;
- inspect table widths, captions, pending-note styling, and print-scale readability;
- inspect the final Git diff and status without committing.

## Scope controls

This work will not execute the final campaign, run expensive training, alter scientific code or frozen experiment parameters, modify historical artifacts, claim deployment readiness, or generate quantitative graphics from invented data.
