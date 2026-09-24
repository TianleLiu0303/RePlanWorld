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
incomplete full `trainval` download.

## Full navtrain token and sensor screen

The complete local `navtrain.yaml` filter contains 1,192 scene names and
103,288 tokens. All scene names matched local sensor directories, and all
103,288 tokens matched exactly one frame in the full trainval pickle logs. No
token was missing or duplicated, and no requested input/future window crossed
a log boundary.

For each token, the screen checked all eight cameras at the four NAVSIM input
offsets (-3, -2, -1, 0) and the ten following offsets (+1 through +10). Every
camera had the same counts:

| Frame offset | Images found / expected | Coverage |
| ---: | ---: | ---: |
| -3 | 96,820 / 103,288 | 93.74% |
| -2 | 98,466 / 103,288 | 95.33% |
| -1 | 100,527 / 103,288 | 97.33% |
| 0 | 103,288 / 103,288 | 100.00% |
| +1 | 88,637 / 103,288 | 85.82% |
| +2 | 77,188 / 103,288 | 74.73% |
| +3 | 67,827 / 103,288 | 65.66% |
| +4 | 60,068 / 103,288 | 58.15% |
| +5 | 53,757 / 103,288 | 52.05% |
| +6 | 48,896 / 103,288 | 47.34% |
| +7 | 45,288 / 103,288 | 43.85% |
| +8 | 42,295 / 103,288 | 40.95% |
| +9 | 39,813 / 103,288 | 38.55% |
| +10 | 37,757 / 103,288 | 36.56% |

Across the four input offsets, 96.60% of camera-frame slots are present. Across
the ten future offsets, 54.36% are present. A direct post-scan check confirmed
that 20 sampled missing history paths remain absent from the mounted
directories; eight sampled current-frame JPEGs, one per camera, decoded
successfully. This points to a `navtrain` sensor subset with complete current
frames but some missing history files and intentionally limited future data.
The official [NAVSIM troubleshooting notes](https://github.com/autonomousvision/navsim/blob/main/docs/splits.md#troubleshooting)
also document missing files in `navtrain` downloads and recommend verifying
the original package checksums. The extracted local data has no source archive
checksums available, so we cannot distinguish missing files in the original
packages from files lost during transfer or storage.

All 774,886 checked frames have fields with the expected pose, command, and
dynamic-state shapes. No frame is missing a timestamp and no adjacent timestamp
pair decreases or repeats. Per-log median intervals are approximately 0.5 s.
These checks do not validate coordinate frames, pose values, camera calibration,
or visual content.

The current runtime cannot contact the NVIDIA driver through `nvidia-smi`.
No model-weight candidates were found under the five configured repository
search roots. This does not establish that weights are absent elsewhere.

**Decision: NO_GO for visual transition training with the current files.**
Current-frame images cover every `navtrain` token, but the four-frame input
history has missing files and future image coverage decreases to 36.56% at
offset +10. Recover or explicitly mask missing history/future targets before
training. Usable GPU access, visual encoder weights, pose/calibration contracts,
and broader visual spot checks also remain open. Mini and trainval share 54 log
filenames, so mini must not be treated as an independent validation split.

Reproduce the checks from the repository root with `make test`,
`make audit-mini-full`, `make audit-trainval-full`, and `make audit-navtrain`.
The filter screen writes [`navtrain_screen.json`](navtrain_screen.json). Full scans save
per-log progress in `/tmp/replanworld_audit_progress`, which was used to resume
after the data mount disconnected during this audit.
