# Phase 0: data and environment audit

The JSON reports in this directory record the exact log sample and coverage.
Read [`summary.md`](summary.md) for the full mini and trainval findings and
the current stage decision.
`data_audit.json` is a bounded smoke audit; `data_audit_full.json` is produced
only by a full run. Neither report is evidence that future RGB is complete
unless its image-path checks cover the required training split and cameras.

Go/No-Go for Phase 0 remains **No-Go** until image availability, timestamp and
pose contracts, visual-encoder weights, and usable training hardware are
confirmed. The data audit is read-only and does not download data.
