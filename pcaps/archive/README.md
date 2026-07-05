# Archived pcaps (deprecated early lab captures)

Early private-lab PCAP captures and their extracted flow CSVs (`deprecated_lab_*`), **superseded** by the operator-generated lab-capture traffic used for the official Phase 2 run (`pcaps/lab_capture_traffic.csv`). Kept tracked for traceability only; no code path references them (verified by grep). See `docs/audits/repo_cleanup_implementation_guide_2026-06-25.md` (B1 / Fase 4).

## Tracking status (2026-07)

The `.pcap`/`.csv` artifacts in this directory are no longer git-tracked
(`git rm --cached`, 2026-07): they remain on this machine's disk and in git
history (decision D-6, no rewrite), but do not ship in fresh clones. This
README stays tracked as the record of what the archive contains.
