# Phase 0 audit summary

Generated 2026-09-24 UTC. These are read-only checks of the local NAVSIM
pickle metadata and the image paths it names. The complete per-log counts are
in [`data_audit_mini_full.json`](data_audit_mini_full.json) and
[`data_audit_trainval_full.json`](data_audit_trainval_full.json).

| Split | Logs | Frames | Images found per camera | Images missing per camera | Logs without sensor directory |
| --- | ---: | ---: | ---: | ---: | ---: |
| mini | 64 | 51,867 | 51,867 | 0 | 0 |
| trainval | 1,310 | 723,019 | 144,954 | 578,065 | 118 |

All eight configured cameras have the same counts. In `trainval`, no log has
complete image coverage: 1,192 have partial coverage and 118 have no images
at their metadata paths. The audit found 1,192 sensor log directories. These
counts describe files available at the configured local paths; they do not
identify whether absent images were never downloaded or are stored elsewhere.

All 774,886 checked frames have fields with the expected pose, command, and
dynamic-state shapes. No frame is missing a timestamp and no adjacent timestamp
pair decreases or repeats. Per-log median intervals are approximately 0.5 s.
These checks do not validate coordinate frames, pose values, camera calibration,
or visual content.

The current runtime cannot contact the NVIDIA driver through `nvidia-smi`.
No model-weight candidates were found under the five configured repository
search roots. This does not establish that weights are absent elsewhere.

**Decision: NO_GO for visual training.** First locate or restore the matching
`trainval` sensor images, confirm usable GPU access and visual encoder weights,
then perform the pose/calibration contract check and visual spot checks. The
mini split is sufficient for loader and preprocessing development while those
prerequisites are resolved. Mini and trainval share 54 log filenames, so mini
must not be treated as an independent validation split.

Reproduce the checks from the repository root with `make test`,
`make audit-mini-full`, and `make audit-trainval-full`. Full scans save
per-log progress in `/tmp/replanworld_audit_progress`, which was used to resume
after the data mount disconnected during this audit.
