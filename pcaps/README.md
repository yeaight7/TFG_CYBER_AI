# `pcaps/` — Phase-2 lab-traffic data & provenance

This directory holds the **Phase-2 evaluation traffic** consumed by
`scripts/predict_real_traffic_v2.py`, plus archived deprecated captures.
This file is the honest provenance record for that data (audit task **C1**).

## Files

| File | Tracked? | Size | Notes |
|------|----------|------|-------|
| `lab_capture_traffic.csv` | gitignored | ~825 MB | 2,000,000 flow rows; the MAIN Phase-2 input. (Renamed from `synthetic_real_traffic.csv` — task C2 / D-7.) |
| `lab_capture_traffic_200k.csv` | gitignored | ~83 MB | 200,000-row subset for quick runs. |
| `archive/deprecated_lab_*.pcap` / `*.csv` | tracked (raw) | ~33 MB | Earlier, deprecated lab captures/flows. See `archive/README.md`. |

## What this data is

It is **real captured network traffic** — genuine packets, not algorithmically
synthesised feature rows — produced in a **closed, isolated home lab** that the
**operator drove by hand (console commands)**: the operator generated both
benign and attack-style traffic between lab hosts, captured the packets,
extracted flow features (CICFlowMeter-py-compatible column names; see
`FLOWMETER_PY_TO_CANON` in `scripts/predict_real_traffic_v2.py:49-126`), and
labelled each flow by the **intent of what was being generated**
(`source_label` → `truth_label` / `truth_y` in the CSV).

Because of this, the appropriate framing — and the framing the docs/thesis must
use — is:

- **Real captured packets** (so "synthetic" is the *wrong* word; the file will be
  renamed `lab_capture_traffic*.csv`).
- **Operator-generated** in a **closed, non-adversarial home lab** — there is no
  real adversary, so it is **not** real-world / production traffic.
- **Labels are trustworthy** (the operator knew what each flow was), but they are
  *generator-intent* labels, **not** an independent detector's verdict.
- **Limited external validity:** a high Phase-2 score is an *in-/near-distribution
  controlled-lab check*, not evidence of real-world detection performance. Every
  Phase-2 claim must cite its `RUN_ID` (the MAIN one is
  `runs/phase2/P2v2_pred_20260610_161231_MAIN/`).

## Note on the GCP/Docker generator (deprecated)

The repo still contains an earlier, **abandoned** attempt at generating Phase-2
traffic with Docker on GCP (`lab/docker/docker-compose.yaml` and
`lab/docker/generator/gen_traffic.py`). Those generated flows turned out to be
**unusable**, so that approach was dropped in favour of the hand-driven home-lab
capture described above. Treat `lab/docker/` and `gen_traffic.py` as deprecated;
they are **not** the source of the committed Phase-2 CSV.

## For full reproducibility (optional, lightweight)

The capture/extraction/labelling was done manually and is intentionally modest
("cutre but honest"). If a fully reproducible record is ever wanted, note: the
capture tool + interface, the exact flow-extractor build, and the
`source_label`-assignment step. This is **not** required for the current
in-distribution-only Phase-2 claim.
