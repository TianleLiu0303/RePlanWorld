# Phase 0 audit summary

Generated 2026-09-24 UTC. These are read-only checks of the local NAVSIM
pickle metadata and the image paths it names. The complete per-log counts are
in [`data_audit_mini_full.json`](data_audit_mini_full.json) and
[`data_audit_trainval_full.json`](data_audit_trainval_full.json).

| Split | Logs | Frames | Images found per camera | Images missing per camera | Logs without sensor directory |
| --- | ---: | ---: | ---: | ---: | ---: |
| mini | 64 | 51,867 | 51,867 | 0 | 0 |
| trainval | 1,310 | 723,019 | 144,954 | 578,065 | 118 |

All eight configured cameras have the same counts. The `trainval` rows above
compare the local sensor files with **every frame** in the full OpenScene
`trainval` metadata. Under that definition, 1,192 logs have partial coverage
and 118 have no images at their metadata paths.

The [NAVSIM split documentation](https://github.com/autonomousvision/navsim/blob/main/docs/splits.md)
explains the distinction: a `navtrain` download contains complete `trainval`
logs but only the sensor frames needed by the filtered `navtrain` split. Full
OpenScene `trainval` sensors exceed 2 TB; the `navtrain` subset is about 445 GB.
The [official full download script](https://github.com/autonomousvision/navsim/blob/main/download/download_trainval.sh)
lists 200 camera archives and 200 LiDAR archives. The local
`sensor_blobs/trainval` directory has exactly the same 1,192 scene names as
the local official `navtrain.yaml` filter, with no extra or missing scene
directory relative to that filter. It also contains all current-frame front
images for the `navtrain` tokens checked in three sample logs. This is strong
evidence that the local sensor set is the intended `navtrain` subset, not an
incomplete full `trainval` download. A full token-and-history coverage check
is still needed to certify the local `navtrain` sensor download.

All 774,886 checked frames have fields with the expected pose, command, and
dynamic-state shapes. No frame is missing a timestamp and no adjacent timestamp
pair decreases or repeats. Per-log median intervals are approximately 0.5 s.
These checks do not validate coordinate frames, pose values, camera calibration,
or visual content.

The current runtime cannot contact the NVIDIA driver through `nvidia-smi`.
No model-weight candidates were found under the five configured repository
search roots. This does not establish that weights are absent elsewhere.

**Decision: NO_GO for training on arbitrary full-`trainval` visual sequences.**
The missing future-frame images are expected when using `navtrain` sensors.
Before choosing the training split, determine whether the RePlanWorld targets
require future RGB at every 2 Hz step. A `navtrain`-filtered input-only baseline
may be viable after its token-and-history coverage is verified. Usable GPU
access, visual encoder weights, pose/calibration contracts, and visual spot
checks also remain open. Mini and trainval share 54 log filenames, so mini
must not be treated as an independent validation split.

Reproduce the checks from the repository root with `make test`,
`make audit-mini-full`, and `make audit-trainval-full`. Full scans save
per-log progress in `/tmp/replanworld_audit_progress`, which was used to resume
after the data mount disconnected during this audit.
