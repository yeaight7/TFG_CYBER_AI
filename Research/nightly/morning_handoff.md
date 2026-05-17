# Morning Handoff
## Second revised pass — 2026-05-16

This handoff supersedes the first-pass version from the same date. It reflects the state of all
five nightly files after two complete passes.

---

## 1. Files Modified in This Session

| File | Status | What changed |
|---|---|---|
| `Research/nightly/state_of_art_expansion.md` | **Rewritten** | Restructured from a "paste instructions" guide into flowing academic prose (Sections A–F) plus a consolidated citation notes table and key-mismatch administrative section. Removed all "Target location" and "Paste this paragraph" meta-commentary from the prose sections. |
| `Research/nightly/research_gap_positioning.md` | **Revised** | Condensed three-section landscape into two focused sections; tightened gap versions A/B/C; cleaned up inline VERIFY markers; retained forbidden-claim table, evidence mapping, and thesis-ready sentences. |
| `Research/nightly/literature_matrix.md` | **Revised** | Added Tier column (Core/Supporting/Context/Future); reorganised 11 clusters; removed redundant Cluster 7 (dataset-as-env) by folding content into Cluster 6; added priority summary table at bottom. |
| `Research/nightly/examiner_risk_review.md` | **Created** | New file: 7 core examiner challenges with honest responses; methodological weakness table; dataset limitation framing; RL vs. supervised objections; CICIDS2017 realism defence; PERMIT/BLOCK justification; contribution defence statement. |
| `Research/nightly/morning_handoff.md` | **This file** | Second-pass complete handoff. |

No source code, experiments, datasets, secrets, CI, or deployment configuration was modified. No
commits were made. No files were deleted.

---

## 2. Best Material Produced

**Most directly useful for thesis drafting:**

- `state_of_art_expansion.md` Sections A–F: six thesis-ready academic paragraphs on CICIDS2017
  quality concerns, canonical schema, named RL-IDS prior works, dataset-as-environment precedent,
  data efficiency, and foundational evaluation limits. Each is labelled with its target draft
  section and has associated citation notes.

- `research_gap_positioning.md` Section 4, Version A (conservative): this is the positioning
  paragraph to use in the thesis introduction and memory chapter. It is cautious, accurate, and
  directly supported by confirmed citations.

- `examiner_risk_review.md` Section 7 (defence statement): the closing paragraph can be used
  verbatim in the conclusion chapter or adapted for the oral defence.

**Most directly useful for bibliography work:**

- `state_of_art_expansion.md` Administrative section: lists all 7 citation key mismatches with
  exact safe replacements. Fix these before any LaTeX build.

- `literature_matrix.md` Priority summary table: lists the citation verification tasks in priority
  order with DOIs for the CANDIDATE entries.

---

## 3. Material That Still Needs Human Review

| Item | Why human review is needed |
|---|---|
| Section A prose (CICIDS2017 quality) | Engelen and Lanvin citations are VERIFY; paragraph should not be in final thesis until those are confirmed |
| Section B prose (canonical schema) | Sarhan citation is VERIFY |
| Section C prose (named RL-IDS works) | Three CANDIDATE citations need DOI verification; ACD paragraph needs a confirmed ACD source |
| Section E prose (data efficiency) | DiMonda citation is VERIFY |
| Section F prose (evaluation limits) | Sommer/Paxson and Axelsson are VERIFY |
| All three gap versions (A/B/C) | Human should pick one and not blend them |
| Examiner responses in `examiner_risk_review.md` | These are draft framings; thesis author should adapt in their own voice |

---

## 4. Missing Citations and Verification Points

### Blocking (fix before LaTeX build)

Run this command to confirm which keys are actually missing from `report/references.bib`:

```bash
grep -c "Ring2017FlowBasedIDS\|CrossDomain2023NIDS\|EvalLongTerm2022\|DatasetSurvey2025\|DLNIDSSurvey2024\|CostSensitiveIDSModel\|CSEIDS2021" report/references.bib
```

Expected result: 0. If any key appears, investigate whether the variant in the draft differs from
a real bib key or whether the bib entry was added after the audit.

Replacements confirmed safe:
- `Ring2017FlowBasedIDS` → `Sperotto2010FlowIDS` (flow background) or `Ring2019DatasetSurvey` (dataset survey); check context per occurrence
- `CrossDomain2023NIDS` → `Layeghy2023CrossDomainNIDS`
- `EvalLongTerm2022NIDS` → `Apruzzese2022CrossEvaluationNIDS`
- `CostSensitiveIDSModel` → `Lee2002CostSensitiveIDS`

### High priority (add to bib before final submission)

| Key | DOI or metadata | Action |
|---|---|---|
| `LopezMartin2020DRLIDS` | 10.1016/j.eswa.2019.112963 | Verify DOI resolves; add full bib entry |
| `LopezMartin2021RBFOfflineRL` | 10.1109/ACCESS.2021.3127689 | Verify DOI; add full bib entry |
| `Ren2022IDRDRL` | 10.1038/s41598-022-19366-3 | Verify DOI; add full bib entry |
| `SommerPaxson2010ClosedWorld` | Likely 10.1109/SP.2010.25 | Confirm DOI; add @inproceedings entry |
| `Axelsson1999BaseRate` | Likely 10.1145/357802.357804 | Confirm DOI; add @article entry |
| `Engelen2021CICIDSIssues` | Unknown | Locate paper; find venue and DOI |
| `Lanvin2023CICIDSFaulty` | Unknown | Locate paper; find venue and DOI |
| `Sharafaldin2018CICIDSAnalysis` | Unknown if in bib | Run: `grep "Sharafaldin2018CICIDSAnalysis" report/references.bib` to confirm |

### Medium priority

| Key | Action |
|---|---|
| `Sarhan2022StandardFeatureSet` | Locate venue and DOI; Sarhan et al. 2022, likely journal |
| `DiMonda2024FewShotNIDS` | Locate venue and DOI; Di Monda et al. approx. 2024 |
| `Cantone2024CrossDataset` | Locate venue and DOI; Cantone et al. approx. 2024 |
| `DatasetSurvey2025NIDS` | Locate or remove; 2025 year suggests instability |
| `DLNIDSSurvey2024` | Locate or use `LiuLang2019IDSSurvey` as interim |

---

## 5. Suggested Next Prompt — Codex / Claude: Integrate SoA into Draft

Use this to integrate the expansion paragraphs into the existing thesis draft:

```
You are working on the repository TFG_CYBER_AI.

Goal:
Integrate the content from Research/nightly/state_of_art_expansion.md into the existing
State of the Art draft at report/drafts/state_of_the_art.md.

Context:
- The expansion file has 6 prose sections (A–F), each labelled with the target location in
  the existing draft.
- Citation keys marked [CANDIDATE] or [VERIFY] must not be silently dropped or altered.
  Keep them with their annotation in the draft text until they are confirmed.
- The existing draft has 7 citation key mismatches documented in the Administrative section
  of the expansion file. Fix these mismatches at the same time.
- Do not rewrite existing prose. Only insert the new paragraphs at the specified locations
  and fix the citation key mismatches.
- Do not alter the structure or section numbering of the existing draft.

Tasks:
1. Read report/drafts/state_of_the_art.md.
2. Read Research/nightly/state_of_art_expansion.md.
3. For each section A–F in the expansion file, insert the corresponding prose at the labelled
   location in the draft.
4. Fix the 7 citation key mismatches listed in the Administrative section using the provided
   safe replacements. Check context per occurrence for Ring2017FlowBasedIDS.
5. Write the updated draft back to report/drafts/state_of_the_art.md.
6. Do not commit.

Constraints:
- Do not invent or modify citation metadata.
- Do not rewrite or paraphrase existing prose.
- Do not remove the [CANDIDATE] or [VERIFY] annotations from inserted paragraphs.
- Do not modify source code, experiments, datasets, or CI configuration.
```

---

## 6. Suggested Next Prompt — Perplexity / Deep Research: Verify Citations

Use this to verify the unconfirmed citation metadata before adding entries to the bib:

```
I am writing a bachelor thesis on RL-based cybersecurity defence for binary PERMIT/BLOCK decisions
on network flows. I need to verify the following academic citations before including them in my
bibliography. For each, please provide: full author list, title, venue (journal or conference),
year, volume/issue/pages if applicable, and DOI if available.

1. Engelen et al., "Troubleshooting an Intrusion Detection Dataset: The CICIDS2017 Case Study,"
   approximate year 2021. I believe the authors include Engelen, Rimmer, and Latré.

2. Lanvin et al., "Faulty use of the CIC-IDS 2017 Dataset in Information Security Research,"
   approximate year 2023.

3. Sarhan et al., "Towards a Standard Feature Set for Network Intrusion Detection System Datasets,"
   approximate year 2022.

4. Di Monda et al., "Few-Shot Class-Incremental Learning for Network Intrusion Detection Systems,"
   approximate year 2024.

5. Cantone et al., "On the Cross-Dataset Generalisation of Machine Learning for Network Intrusion
   Detection," approximate year 2024.

6. Robin Sommer and Vern Paxson, "Outside the Closed World: On Using Machine Learning for Network
   Intrusion Detection," IEEE Symposium on Security and Privacy 2010. Please confirm DOI.

7. Stefan Axelsson, "The Base-Rate Fallacy and the Difficulty of Intrusion Detection," ACM
   Transactions on Information and System Security, vol. 3, no. 3, 2000. Please confirm DOI.

8. López-Martín et al., "Application of Deep Reinforcement Learning to Intrusion Detection for
   Supervised Problems," Expert Systems with Applications, 2020. DOI: 10.1016/j.eswa.2019.112963.
   Please confirm this DOI is correct.

For each citation, mark any field you cannot confirm. Do not invent metadata.
```

---

## 7. Git Commands to Inspect Changes

Run these to verify what was changed in this session:

```bash
# See all untracked files (the four nightly files)
git status

# See the content of the new examiner review file
git diff HEAD -- "Research/nightly/examiner_risk_review.md"

# Diff against the previous committed state for all nightly files
git diff HEAD -- "Research/nightly/"

# Show when the nightly files were last modified
git log --oneline --all -- "Research/nightly/"

# If you want to review just the new prose sections in the expansion file
git diff HEAD -- "Research/nightly/state_of_art_expansion.md" | head -150
```

Note: since these files were never committed, `git diff HEAD` will show nothing for them. Use
`git status` to confirm they are untracked, and read them directly to review content.

---

## 8. State of All Nightly Files

| File | Status | Ready to use? |
|---|---|---|
| `state_of_art_expansion.md` | Rewritten: 6 prose sections + citation notes + admin section | Yes — after fixing citation keys and verifying VERIFY citations |
| `research_gap_positioning.md` | Revised: tighter gap versions, clean evidence mapping | Yes — use Version A (conservative) for introduction |
| `literature_matrix.md` | Revised: added Tier column, priority summary | Yes — use as citation planning reference |
| `examiner_risk_review.md` | Created: 7 challenges + tables + defence statement | Yes — read before writing conclusion and before oral defence |
| `morning_handoff.md` | This file | — |

**Confidence level on content quality:** High for confirmed-citation content. Medium for content
that relies on VERIFY citations. Low for any claim that requires RF or leave-one-out artifacts
that do not yet exist.

**Most urgent action before writing the next thesis chapter:** fix the 7 citation key mismatches
in `report/drafts/state_of_the_art.md`. Until those are fixed, the draft will not compile.
