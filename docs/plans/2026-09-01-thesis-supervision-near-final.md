# Thesis supervision near-final implementation plan

Date: 2026-09-01  
Branch: `chore/yeaight7/thesis-supervision-near-final-pass`

## Task 1: Establish the evidence boundary

- Update `memoria/capitulos/resumen.tex` and `memoria/capitulos/introduccion.tex` so stable contributions do not depend on pending campaign performance.
- Update objective/status wording in `memoria/capitulos/objetivos_y_alcance.tex`.
- Add a professional pending-evidence note in `memoria/memoria.tex` and retire visible pending boxes from maintained content.

## Task 2: Synchronize methods with the approved campaign

- Correct current semantics and implementation details in `memoria/capitulos/diseno_sistema.tex`.
- Rewrite `memoria/capitulos/protocolo_experimental.tex` around the frozen `main-v1` matrix in `experiments/final_experiment_campaign.json`.
- Align appendices `A`, `B`, and `D` with the current preflight, cache, seeds, schema, export, LFS, and targeted-holdout contracts.
- Use provider-neutral final-platform language and preserve hardware facts only inside explicitly historical evidence.

## Task 3: Restructure evidence, discussion, and conclusions

- Rewrite `memoria/capitulos/resultados.tex` with a final-campaign insertion hierarchy and a bounded historical-results section.
- Rewrite `memoria/capitulos/discusion.tex` as thematic analysis prepared for final evidence.
- Tighten `memoria/capitulos/limitaciones.tex` and remove redundant draft placeholders.
- Rewrite `memoria/capitulos/conclusiones.tex` as a coherent provisional conclusion with supported and conditional claims.

## Task 4: Improve figures and figure tooling

- Create `docs/thesis/FIGURE_AUDIT.md` with one decision per existing and planned figure.
- Improve the stale campaign/validation/Phase 2 conceptual diagrams and add campaign-overview and traceability diagrams where they replace prose.
- Remove unused synthetic non-empirical learning-curve assets.
- Add `scripts/make_final_campaign_thesis_figures.py`, which validates final artifacts before rendering exact quantitative figures.
- Keep quantitative historical figures artifact-backed and label their evidence status in captions.

## Task 5: Clean bibliography and supporting register

- Correct verified metadata in `memoria/memoria.bib`.
- Remove unverifiable draft citations from maintained prose rather than inventing metadata.
- Create `docs/thesis/FINALIZATION_REGISTER.md` as the short operational answer to “what remains before submission?”.

## Task 6: Build and inspect

- Run `latexmk -C` and a complete `latexmk -pdf` build from `memoria/`.
- Check the log for undefined citations/references and material overfull boxes.
- Check PDF metadata, page count, contents, list of figures, and list of tables.
- Render representative pages from every chapter and every changed or added figure with Poppler.
- Inspect the rendered images at thesis scale and correct layout issues.

## Task 7: Review the deliverable

- Run `git diff --check`, inspect `git diff --stat`, and review every changed file.
- Confirm no experiment artifact, scientific implementation, hyperparameter, reward, or feature contract changed.
- Report commands, validation, exact changed files, residual pending evidence, refused claims, and the recommended artifact-ingestion next step.
