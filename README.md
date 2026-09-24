# RePlanWorld

RePlanWorld is a staged research project for replanning-consistent visual
dynamics on NAVSIM. The repository currently contains the Phase 0 project
foundation and a **read-only** local data and environment audit. Model training
has not started.

## Phase 0: audit local data

The audit reads the existing NAVSIM pickle logs and checks camera image paths.
It does not copy, change, or download any data. Raw NAVSIM pickles require a
Python environment with NumPy; the provided `lingbot-map` environment can load
them on this machine. The tests use Python's standard library.

Run these commands from the repository root. `make` resolves the local
workspace and the bundled `lingbot-map` Python environment from the repository
location, so an absolute mount path is not embedded in the audit command.

```bash
make test
make audit-smoke
```

The smoke audit selects eight evenly spaced logs from each configured split,
checks every frame in those logs, and writes
[`reports/phase0/data_audit.json`](reports/phase0/data_audit.json). Its image
coverage numbers describe **only the sampled logs**. Run a full audit when the
data inventory and runtime permit it:

```bash
make audit-full
```

For a complete check of the smaller `mini` split alone, run
`make audit-mini-full`. To check every `trainval` log, run
`make audit-trainval-full`. Full scans save resumable progress in `/tmp`;
the cache rechecks log and sensor directory modification times before reuse.
The all-split full run writes `reports/phase0/data_audit_full.json`. Configure other
data locations and camera sets in [`configs/audit.json`](configs/audit.json), or
pass a different `--workspace-root` to `python -m replan_world.data.audit`.
The raw logs are trusted local pickle files; do not run this command on
untrusted pickle files.

Each report records source directory inventory, selected log names, frame
fields, timestamp intervals, pose/command availability, per-camera image path
counts, cross-dataset log overlap, GPU visibility, installed package versions, code revisions, and
checkpoint candidates in the configured search roots. A `NO_GO` decision
means the next visual-training phase cannot yet be justified. A sampled audit
does not certify the complete training split.

The full-split findings and current gate are summarized in
[`reports/phase0/summary.md`](reports/phase0/summary.md).

## Planned implementation order

1. Confirm the Phase 0 data/time/pose contract and available training hardware.
2. Adapt DriveDreamer-Policy's one-scene-per-pickle metadata preprocessing.
3. Build a visual Action-only NAVSIM baseline using the WA-JEPA agent interface.
4. Train and validate Change tokens and a transition decoder.
5. Validate an independent local Replanner and an oracle planner.
6. Train matched Change–Action baselines, then add RPE and the 2×2 ablation.

Each stage requires tests, a reproducible command, a report, and a Go/No-Go
decision before the next stage begins. See
[`reports/phase0/README.md`](reports/phase0/README.md) for the current gate.
