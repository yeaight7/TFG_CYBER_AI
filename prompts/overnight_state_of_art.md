You are working on the repository `TFG_CYBER_AI`.

Goal:
Expand and strengthen the thesis state-of-the-art material for an academic bachelor thesis about an RL-based cybersecurity defender for binary PERMIT/BLOCK decisions on network flows.

Context:
- Main thesis positioning:
  - RL-based cybersecurity defender.
  - Public dataset benchmark, mainly CICIDS2017.
  - Canonical flow-feature preprocessing.
  - Gymnasium dataset-as-environment formulation.
  - QRDQN agent.
  - Binary action space: PERMIT/BLOCK.
  - Cost-sensitive concern: false negatives are more dangerous than false positives.
  - Supervised baseline planned, at least Random Forest.
  - Data-efficiency experiments planned: 100k/250k/500k/1M/2M train sizes with same internal test.
  - External validation using privately captured lab traffic may be included only if viable.
  - No claim of active real-time network blocking.
  - Do not claim the thesis is completely novel.
  - Do not claim RL for IDS has not been studied.
  - Do not claim production readiness.
  - Do not claim real-world deployment.

Primary task:
Inspect the current repository, especially:
- Research/
- docs/
- thesis/report/memory-related files if present
- bibliography files if present
- README and experiment documentation if useful

Then produce or update state-of-the-art material.

Allowed changes:
- Create or update files under `Research/`, `docs/`, or thesis draft directories if they already exist.
- Create a file `Research/nightly/state_of_art_expansion.md`.
- Create a file `Research/nightly/literature_matrix.md`.
- Create a file `Research/nightly/research_gap_positioning.md`.
- Create a file `Research/nightly/morning_handoff.md`.

Do not:
- Push to remote.
- Commit unless explicitly asked later.
- Delete files.
- Rewrite git history.
- Modify source code.
- Modify experiments.
- Modify datasets.
- Modify secrets, credentials, environment files, CI, or deployment configuration.
- Invent citations.
- Overclaim novelty.

Procedure:
1. First inspect the existing research material and summarize what already exists.
2. Identify gaps in the state-of-the-art draft.
3. Expand the state of the art around:
   - ML/DL for NIDS.
   - RL/DRL for IDS and cyber defense.
   - Dataset-as-environment formulations.
   - Binary flow-level PERMIT/BLOCK framing.
   - Cost-sensitive evaluation and false-negative risk.
   - CICIDS2017 and common NIDS benchmark concerns.
   - Reproducibility and leakage-aware experimental design.
   - Positioning against supervised baselines.
4. Build a literature matrix with columns:
   - Topic
   - Work / source
   - What it contributes
   - Relevance to this thesis
   - Limitation / gap
   - How to cite or verify
5. Draft a defensible research gap and thesis positioning.
6. Produce a morning handoff with:
   - Files changed
   - Main claims added
   - Claims that still need verification
   - Suggested next prompt for the next agent
   - Open risks

Citation discipline:
- If exact citation metadata is already available in the repo, use it.
- If citation metadata is not available, mark it as `VERIFY_CITATION`.
- Do not fabricate DOI, year, venue, author names, or BibTeX entries.
- Prefer cautious wording.

Writing style:
- English.
- Academic but direct.
- No marketing language.
- No placeholders unless unavoidable.
- No Spanish unless existing files are already Spanish.

Done when:
- The four files under `Research/nightly/` exist.
- The state-of-the-art expansion is useful enough to paste into the thesis draft after human review.
- The handoff clearly explains what was done and what remains uncertain.
