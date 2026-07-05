# Research Notes Index

Consolidated research material (2026-07 reorganisation: formerly the root
`Research/` tree and `docs/Personal Research/`). These are **research notes,
NOT a source of truth** — authoritative claims live in
[../results.md](../results.md) and the run artifacts under `runs/`.

| Subtree | Contents | Language |
|---------|----------|----------|
| [state-of-the-art/](state-of-the-art/) | Normalized literature base for the State-of-the-Art chapter (start at its `README.md` / `RESEARCH_INDEX.md`); includes `nightly/` agent handoffs. Feeds the `report/` EN chapters. | English |
| [state-of-the-art-raw-dumps/](state-of-the-art-raw-dumps/) | Four raw ChatGPT/Perplexity deep-research exports, kept as received. | English |
| [personal/](personal/) | Personal deep-research notes, including [personal/deep-defense-research/](personal/deep-defense-research/) (7-part Spanish tribunal-prep package). | Spanish/English |

Caveats:

- The raw dumps are as-received artifacts: they contain two dangling
  `references.bib` links and stale internal path mentions retained verbatim
  as historical research material. One export had an expired AWS presigned
  URL redacted on 2026-07-05 (see the note inside the file).
- Pre-2026-07 documents (e.g. the audits under [../audits/](../audits/))
  reference these files at their old paths (`Research/…`,
  `docs/Personal Research/…`); use `git log --follow` to trace moves.
