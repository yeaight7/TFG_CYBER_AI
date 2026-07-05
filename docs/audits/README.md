# docs/audits — auditorías del repositorio

Auditorías read-only datadas. **No** son fuente de verdad operativa: son fotos del estado en su fecha. La fuente de verdad es el código + artefactos en `runs/`, y los docs mantenidos en `docs/`.

| Documento | Fecha | Alcance | Estado |
|-----------|-------|---------|--------|
| [repo_cleanup_implementation_guide_2026-06-25.md](repo_cleanup_implementation_guide_2026-06-25.md) | 2026-06-25 | Limpieza/realineamiento por fases (consolidó los audits previos) | Histórica — superada por la auditoría del 27-06 y la reorganización 2026-07 |
| [REPO_AUDIT_2026-06-27.md](REPO_AUDIT_2026-06-27.md) | 2026-06-27 | Auditoría integral de estado (código, docs, artefactos, seguridad, deps) | Histórica — con notas de corrección 2026-07-05 |
| [AUDIT_REMEDIATION_PLAN.md](AUDIT_REMEDIATION_PLAN.md) | 2026-06-27/28 | Plan/tracker de remediación + registro de decisiones D-1..D-11 | Histórico — ejecutado casi por completo; ítems de tooling obsoletos tras la limpieza 2026-07 |

Notas:

- Los audits previos `tfg_cyber_ai_audit.md` y `stale_claims_diagnosis_2026-06-15.md` se consolidaron en la guía del 25-06 (§11) y se **retiraron** el 2026-06-25; quedan en la historia de git.
- Las rutas citadas en cada auditoría reflejan el árbol del repo en su fecha; la reorganización de 2026-07 movió los docs de investigación (`docs/research/`), defensa (`docs/defensa/`) y estas mismas auditorías (antes en la raíz). Usa `git log --follow` para rastrear movimientos.
- El registro de decisiones del propietario (D-1..D-11: sin reescritura de historia, PDFs tracked, sin re-entrenos, etc.) vive en `AUDIT_REMEDIATION_PLAN.md` y sigue siendo vinculante.
