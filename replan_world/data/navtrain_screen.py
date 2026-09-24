"""Check NAVSIM navtrain sensor coverage for inputs and visual futures.

The official navtrain scene filter uses four history/current frames followed
by ten future frames. This screen reports the two ranges independently.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import errno
import json
import os
from pathlib import Path, PurePosixPath
import pickle
import sys
from typing import Any


CAMERAS = ["CAM_F0", "CAM_L0", "CAM_R0", "CAM_L1", "CAM_R1", "CAM_L2", "CAM_R2", "CAM_B0"]
INPUT_OFFSETS = (-3, -2, -1, 0)
FUTURE_OFFSETS = tuple(range(1, 11))


def read_filter(path: Path) -> tuple[list[str], set[str]]:
    """Read log_names and tokens from NAVSIM's simple YAML lists."""
    log_names: list[str] = []
    tokens: set[str] = set()
    section: str | None = None
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("log_names:"):
                section = "logs"
                continue
            if line.startswith("tokens:"):
                section = "tokens"
                continue
            if section and line.startswith("  - '"):
                value = line.split("'", 2)[1]
                if section == "logs":
                    log_names.append(value)
                else:
                    tokens.add(value)
    if not log_names or not tokens:
        raise ValueError(f"No log_names or tokens found in filter: {path}")
    if len(set(log_names)) != len(log_names):
        raise ValueError("Duplicate log names in navtrain filter")
    return log_names, tokens


def _fingerprint(log_path: Path, sensor_root: Path, filter_path: Path) -> dict[str, Any]:
    info = log_path.stat()
    filter_info = filter_path.stat()
    dir_mtimes: dict[str, int | None] = {}
    for camera in CAMERAS:
        path = sensor_root / log_path.stem / camera
        try:
            dir_mtimes[camera] = path.stat().st_mtime_ns
        except FileNotFoundError:
            dir_mtimes[camera] = None
    return {
        "log_size": info.st_size,
        "log_mtime_ns": info.st_mtime_ns,
        "filter_size": filter_info.st_size,
        "filter_mtime_ns": filter_info.st_mtime_ns,
        "camera_dir_mtimes_ns": dir_mtimes,
    }


def _camera_files(sensor_root: Path, log_name: str) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for camera in CAMERAS:
        directory = sensor_root / log_name / camera
        try:
            with os.scandir(directory) as entries:
                result[camera] = {entry.name for entry in entries if entry.is_file()}
        except (FileNotFoundError, NotADirectoryError):
            result[camera] = set()
        except OSError:
            raise
    return result


def scan_log(log_path: Path, sensor_root: Path, filter_tokens: set[str]) -> dict[str, Any]:
    with log_path.open("rb") as stream:
        frames = pickle.load(stream)
    if not isinstance(frames, list) or not frames:
        raise ValueError("Expected non-empty list of frames")
    token_indices: dict[str, int] = {}
    duplicate_frame_tokens = 0
    for index, frame in enumerate(frames):
        token = frame.get("token") if isinstance(frame, dict) else None
        if not isinstance(token, str):
            continue
        if token in token_indices:
            duplicate_frame_tokens += 1
        token_indices[token] = index

    selected_tokens = [token for token in filter_tokens if token in token_indices]
    missing_tokens = [token for token in filter_tokens if token not in token_indices]
    file_names = _camera_files(sensor_root, log_path.stem)
    offsets = INPUT_OFFSETS + FUTURE_OFFSETS
    counts = {
        camera: {str(offset): Counter({"checked": 0, "found": 0, "missing": 0,
                                      "metadata_missing": 0, "invalid_path": 0})
                 for offset in offsets}
        for camera in CAMERAS
    }
    examples: dict[str, list[dict[str, Any]]] = {str(offset): [] for offset in offsets}
    boundary_errors = Counter()
    token_checks: dict[str, int] = Counter()

    for token in selected_tokens:
        center = token_indices[token]
        token_checks[str(center)] += 1
        for offset in offsets:
            frame_index = center + offset
            if frame_index < 0 or frame_index >= len(frames):
                boundary_errors[str(offset)] += 1
                continue
            frame = frames[frame_index]
            cams = frame.get("cams") or {}
            for camera in CAMERAS:
                counter = counts[camera][str(offset)]
                counter["checked"] += 1
                metadata = cams.get(camera)
                data_path = metadata.get("data_path") if isinstance(metadata, dict) else None
                if not isinstance(data_path, str) or not data_path:
                    counter["metadata_missing"] += 1
                    continue
                relative = PurePosixPath(data_path)
                if relative.is_absolute() or ".." in relative.parts:
                    counter["invalid_path"] += 1
                    continue
                expected_parent = (log_path.stem, camera)
                if tuple(relative.parts[-3:-1]) != expected_parent:
                    counter["invalid_path"] += 1
                    continue
                if relative.name in file_names[camera]:
                    counter["found"] += 1
                else:
                    counter["missing"] += 1
                    if len(examples[str(offset)]) < 20:
                        examples[str(offset)].append({
                            "token": token,
                            "frame_index": frame_index,
                            "camera": camera,
                            "relative_path": data_path,
                        })

    return {
        "log_file": log_path.name,
        "frame_count": len(frames),
        "matched_filter_tokens": len(selected_tokens),
        "matched_token_ids": selected_tokens,
        "duplicate_frame_tokens": duplicate_frame_tokens,
        "window_boundary_errors": dict(boundary_errors),
        "camera_offsets": {camera: {offset: dict(counter) for offset, counter in values.items()}
                           for camera, values in counts.items()},
        "missing_examples": examples,
    }


def run_screen(
    workspace_root: Path,
    filter_path: Path,
    checkpoint_path: Path,
    resume: bool,
    trust_cache: bool = False,
) -> dict[str, Any]:
    log_dir = workspace_root / "data_aliyun/navsim_logs/trainval"
    sensor_root = workspace_root / "data_aliyun/sensor_blobs/trainval"
    log_names, tokens = read_filter(filter_path)
    cache: dict[str, dict[str, Any]] = {}
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if resume and checkpoint_path.is_file():
        with checkpoint_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                try:
                    item = json.loads(line)
                    cache[item["log_file"]] = item
                except (json.JSONDecodeError, KeyError):
                    continue
    else:
        checkpoint_path.write_text("", encoding="utf-8")

    per_log: list[dict[str, Any]] = []
    missing_examples: dict[str, list[dict[str, Any]]] = {
        str(offset): [] for offset in INPUT_OFFSETS + FUTURE_OFFSETS
    }
    total_offsets = {camera: {str(offset): Counter() for offset in INPUT_OFFSETS + FUTURE_OFFSETS}
                     for camera in CAMERAS}
    matched_tokens: set[str] = set()
    duplicate_token_matches = 0
    total_boundary_errors: Counter[str] = Counter()
    cache_hits = 0

    for position, name in enumerate(log_names, start=1):
        log_path = log_dir / f"{name}.pkl"
        cached = cache.get(log_path.name)
        if cached and trust_cache:
            result = cached["result"]
            cache_hits += 1
        else:
            fingerprint = _fingerprint(log_path, sensor_root, filter_path)
            if cached and cached.get("fingerprint") == fingerprint:
                result = cached["result"]
                cache_hits += 1
            else:
                result = scan_log(log_path, sensor_root, tokens)
                with checkpoint_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"log_file": log_path.name,
                                             "fingerprint": fingerprint,
                                             "result": result}) + "\n")
        for offset, examples in result["missing_examples"].items():
            available = 20 - len(missing_examples[offset])
            if available > 0:
                missing_examples[offset].extend(examples[:available])
        public_result = {key: value for key, value in result.items()
                         if key not in {"matched_token_ids", "missing_examples", "camera_offsets"}}
        camera_counts = result["camera_offsets"]
        front_counts = camera_counts["CAM_F0"]
        if all(camera_counts[camera] == front_counts for camera in CAMERAS):
            public_result["camera_offsets"] = {"all_cameras_identical": True,
                                                "counts": front_counts}
        else:
            public_result["camera_offsets"] = {"all_cameras_identical": False,
                                                "counts_by_camera": camera_counts}
        per_log.append(public_result)
        previous_matched = len(matched_tokens)
        matched_tokens.update(result["matched_token_ids"])
        duplicate_token_matches += result["matched_filter_tokens"] - (len(matched_tokens) - previous_matched)
        total_boundary_errors.update(result["window_boundary_errors"])
        for camera in CAMERAS:
            for offset, counts in result["camera_offsets"][camera].items():
                total_offsets[camera][offset].update(counts)
        if position == 1 or position % 20 == 0 or position == len(log_names):
            print(f"Screening navtrain sensor coverage: {position}/{len(log_names)} logs; "
                  f"{len(matched_tokens)}/{len(tokens)} tokens matched", file=sys.stderr, flush=True)

    camera_totals = {camera: {offset: dict(counter) for offset, counter in offsets.items()}
                     for camera, offsets in total_offsets.items()}
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "workspace_root": str(workspace_root),
        "filter_file": str(filter_path),
        "filter_log_count": len(log_names),
        "filter_token_count": len(tokens),
        "history_frame_count": len(INPUT_OFFSETS),
        "future_frame_count": len(FUTURE_OFFSETS),
        "camera_names": CAMERAS,
        "coverage": {
            "scanned_log_count": len(per_log),
            "resumed_log_count": cache_hits,
            "matched_filter_tokens": len(matched_tokens),
            "missing_filter_tokens": len(tokens - matched_tokens),
            "duplicate_token_matches_across_logs": duplicate_token_matches,
            "boundary_errors_by_offset": dict(total_boundary_errors),
        },
        "camera_offset_totals": camera_totals,
        "missing_path_examples_by_offset": missing_examples,
        "logs": per_log,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=Path("/data/chaosheng"))
    parser.add_argument("--filter", type=Path, default=Path(
        "navsim/navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml"))
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("/tmp/replanworld_navtrain_screen/progress.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("reports/phase0/navtrain_screen.json"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--trust-cache", action="store_true",
                        help="Reuse existing per-log results without stat checks; use only when source data is unchanged")
    args = parser.parse_args()
    report = run_screen(args.workspace_root.resolve(),
                        args.filter if args.filter.is_absolute() else args.workspace_root / args.filter,
                        args.checkpoint, args.resume, args.trust_cache)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(f"Wrote {args.output}")
    for camera in CAMERAS:
        print(camera)
        for offset, counts in report["camera_offset_totals"][camera].items():
            print(f"  offset {offset:>3}: {counts['found']}/{counts['checked']} found, "
                  f"{counts['missing']} missing")


if __name__ == "__main__":
    main()
