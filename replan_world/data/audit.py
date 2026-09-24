"""Read-only audit of local NAVSIM logs, sensor paths, and runtime.

NAVSIM log pickles are trusted local input. They contain NumPy objects, so the
interpreter used for a real audit must have NumPy installed.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import errno
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import pickle
import platform
import statistics
import subprocess
import sys
from typing import Any


CHECKPOINT_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors"}
SKIP_CHECKPOINT_DIRS = {
    ".git", ".venv", "__pycache__", "navsim", "navsim_v1", "navsim_v1.1",
    "navsim_v2", "vjepa2", "data", "dataset", "cache", "artifacts", "runs",
}


def select_log_files(files: list[Path], limit: int) -> list[Path]:
    """Select evenly spaced logs; always include both ends when possible."""
    ordered = sorted(files)
    if limit <= 0 or limit >= len(ordered):
        return ordered
    if limit == 1:
        return ordered[:1]
    positions = [round(i * (len(ordered) - 1) / (limit - 1)) for i in range(limit)]
    return [ordered[i] for i in positions]


def safe_sensor_path(sensor_root: Path, data_path: Any) -> Path | None:
    if not isinstance(data_path, str) or not data_path:
        return None
    relative = PurePosixPath(data_path)
    if relative.is_absolute() or ".." in relative.parts:
        return None
    return sensor_root.joinpath(*relative.parts)


def _sequence_length(value: Any) -> int:
    try:
        return len(value)
    except (TypeError, ValueError):
        return -1


def _valid_command(value: Any) -> bool:
    if _sequence_length(value) != 4:
        return False
    try:
        values = [float(x) for x in value]
    except (TypeError, ValueError):
        return False
    return all(x == x and abs(x) != float("inf") for x in values)


def _valid_pose(frame: dict[str, Any]) -> bool:
    return (
        _sequence_length(frame.get("ego2global_translation")) >= 2
        and _sequence_length(frame.get("ego2global_rotation")) == 4
    )


def _timestamps(frames: list[dict[str, Any]], unit: str) -> dict[str, Any]:
    values: list[int] = []
    missing = 0
    for frame in frames:
        value = frame.get("timestamp")
        if isinstance(value, int):
            values.append(value)
        else:
            missing += 1
    deltas = [b - a for a, b in zip(values, values[1:])]
    positive = [value for value in deltas if value > 0]
    scale = 1_000_000 if unit == "microseconds" else 1
    return {
        "missing": missing,
        "non_increasing_pairs": len(deltas) - len(positive),
        "first": values[0] if values else None,
        "last": values[-1] if values else None,
        "median_interval_seconds": round(statistics.median(positive) / scale, 6) if positive else None,
        "min_interval_seconds": round(min(positive) / scale, 6) if positive else None,
        "max_interval_seconds": round(max(positive) / scale, 6) if positive else None,
    }


def scan_log(
    log_path: Path,
    sensor_root: Path,
    sensor_log_dirs: set[str],
    cameras: list[str],
    timestamp_unit: str,
    frame_stride: int = 1,
) -> dict[str, Any]:
    with log_path.open("rb") as stream:
        frames = pickle.load(stream)
    if not isinstance(frames, list) or not frames:
        raise ValueError("Expected a non-empty list of NAVSIM frames")
    if not all(isinstance(frame, dict) for frame in frames):
        raise ValueError("Every NAVSIM frame must be a dictionary")

    selected = frames[::frame_stride]
    camera_counts = {
        camera: Counter({"checked": 0, "metadata_missing": 0, "invalid_path": 0,
                         "file_missing": 0, "file_found": 0})
        for camera in cameras
    }
    missing_examples: list[dict[str, Any]] = []
    sensor_log_directory_present = log_path.stem in sensor_log_dirs
    directory_files: dict[Path, set[str]] = {}
    for index in range(0, len(frames), frame_stride):
        frame = frames[index]
        frame_cameras = frame.get("cams") or {}
        for camera in cameras:
            counts = camera_counts[camera]
            counts["checked"] += 1
            metadata = frame_cameras.get(camera)
            if not isinstance(metadata, dict) or not metadata:
                counts["metadata_missing"] += 1
                continue
            data_path = metadata.get("data_path")
            image_path = safe_sensor_path(sensor_root, data_path)
            if image_path is None:
                counts["invalid_path"] += 1
                continue
            parent = image_path.parent
            if sensor_log_directory_present and parent not in directory_files:
                try:
                    directory_files[parent] = {
                        entry.name for entry in os.scandir(parent) if entry.is_file()
                    }
                except (FileNotFoundError, NotADirectoryError):
                    directory_files[parent] = set()
            if sensor_log_directory_present and image_path.name in directory_files.get(parent, set()):
                counts["file_found"] += 1
            else:
                counts["file_missing"] += 1
                if len(missing_examples) < 12:
                    missing_examples.append({
                        "frame_index": index,
                        "camera": camera,
                        "relative_path": data_path,
                    })

    return {
        "log_file": log_path.name,
        "frame_count": len(frames),
        "checked_frame_count": len(selected),
        "sensor_log_directory_present": sensor_log_directory_present,
        "frame_keys": sorted(frames[0].keys()),
        "timestamps": _timestamps(frames, timestamp_unit),
        "pose_valid_frames": sum(_valid_pose(frame) for frame in frames),
        "command_valid_frames": sum(_valid_command(frame.get("driving_command")) for frame in frames),
        "ego_dynamic_state_valid_frames": sum(
            _sequence_length(frame.get("ego_dynamic_state")) == 4 for frame in frames
        ),
        "camera_counts": {camera: dict(counts) for camera, counts in camera_counts.items()},
        "missing_image_examples": missing_examples,
    }


def audit_dataset(
    name: str,
    log_dir: Path,
    sensor_dir: Path,
    cameras: list[str],
    timestamp_unit: str,
    max_logs: int,
    frame_stride: int,
    checkpoint_path: Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    if not log_dir.is_dir() or not sensor_dir.is_dir():
        return {
            "name": name, "status": "missing_directory",
            "log_dir": str(log_dir), "sensor_dir": str(sensor_dir),
        }
    log_files = sorted(path for path in log_dir.iterdir() if path.is_file() and path.suffix == ".pkl")
    sensor_log_dirs = {path.name for path in sensor_dir.iterdir() if path.is_dir()}
    selected_logs = select_log_files(log_files, max_logs)
    cached: dict[str, dict[str, Any]] = {}
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        if resume and checkpoint_path.is_file():
            with checkpoint_path.open("r", encoding="utf-8") as stream:
                for line in stream:
                    try:
                        item = json.loads(line)
                        cached[item["log_file"]] = item
                    except (json.JSONDecodeError, KeyError):
                        continue
        else:
            checkpoint_path.write_text("", encoding="utf-8")
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    totals = {camera: Counter() for camera in cameras}
    cache_hits = 0
    for position, log_path in enumerate(selected_logs, start=1):
        if len(selected_logs) <= 50 or position == 1 or position % 10 == 0 or position == len(selected_logs):
            print(f"Scanning {name}: log {position}/{len(selected_logs)} {log_path.name}",
                  file=sys.stderr, flush=True)
        try:
            fingerprint = _log_fingerprint(log_path, sensor_dir, cameras, timestamp_unit, frame_stride)
            item = cached.get(log_path.name)
            if item is not None and item.get("fingerprint") == fingerprint:
                result = item["result"]
                cache_hits += 1
            else:
                result = scan_log(
                    log_path, sensor_dir, sensor_log_dirs, cameras, timestamp_unit, frame_stride
                )
                if checkpoint_path is not None:
                    with checkpoint_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps({
                            "log_file": log_path.name,
                            "fingerprint": fingerprint,
                            "result": result,
                        }, ensure_ascii=False) + "\n")
        except (OSError, pickle.UnpicklingError, ImportError, ValueError, TypeError) as error:
            if isinstance(error, OSError) and error.errno in {
                errno.ENOTCONN, errno.EIO, errno.ESTALE, errno.ENOENT,
            }:
                try:
                    mount_available = log_dir.is_dir() and sensor_dir.is_dir()
                except OSError:
                    mount_available = False
                if error.errno != errno.ENOENT or not mount_available:
                    raise RuntimeError(
                        f"Workspace mount became unavailable; resume from {checkpoint_path}"
                    ) from error
            errors.append({"log_file": log_path.name, "error": f"{type(error).__name__}: {error}"})
            continue
        results.append(result)
        for camera in cameras:
            totals[camera].update(result["camera_counts"][camera])

    return {
        "name": name,
        "status": "ok" if not errors else "scan_errors",
        "log_dir": str(log_dir),
        "sensor_dir": str(sensor_dir),
        "inventory": {
            "log_pickle_files": len(log_files),
            "sensor_log_directories": len(sensor_log_dirs),
            "logs_with_sensor_directory": len({path.stem for path in log_files} & sensor_log_dirs),
            "logs_without_sensor_directory": len({path.stem for path in log_files} - sensor_log_dirs),
        },
        "coverage": {
            "mode": "full" if len(selected_logs) == len(log_files) and frame_stride == 1 else "sampled",
            "selected_log_count": len(selected_logs),
            "scanned_log_count": len(results),
            "resumed_log_count": cache_hits,
            "frame_stride": frame_stride,
            "selection": "evenly_spaced_sorted_log_filenames",
        },
        "camera_totals_checked": {camera: dict(counts) for camera, counts in totals.items()},
        "logs": results,
        "errors": errors,
    }


def _log_fingerprint(
    log_path: Path, sensor_dir: Path, cameras: list[str], timestamp_unit: str, frame_stride: int
) -> dict[str, Any]:
    info = log_path.stat()
    camera_dir_mtimes: dict[str, int | None] = {}
    for camera in cameras:
        try:
            camera_dir_mtimes[camera] = (sensor_dir / log_path.stem / camera).stat().st_mtime_ns
        except FileNotFoundError:
            camera_dir_mtimes[camera] = None
    return {
        "log_size": info.st_size,
        "log_mtime_ns": info.st_mtime_ns,
        "camera_dir_mtimes_ns": camera_dir_mtimes,
        "cameras": cameras,
        "timestamp_unit": timestamp_unit,
        "frame_stride": frame_stride,
    }


def _command_output(command: list[str], timeout: int = 12) -> dict[str, Any]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"available": False, "error": f"{type(error).__name__}: {error}"}
    return {
        "available": result.returncode == 0,
        "returncode": result.returncode,
        "output": (result.stdout or result.stderr).strip()[:1000],
    }


def _git_revision(path: Path) -> str | None:
    result = _command_output(["git", "-C", str(path), "rev-parse", "--short", "HEAD"], timeout=8)
    return result.get("output") if result.get("available") else None


def _checkpoint_inventory(workspace_root: Path, roots: list[str]) -> dict[str, Any]:
    found: list[str] = []
    searched: list[str] = []
    errors: list[str] = []
    for name in roots:
        root = workspace_root / name
        try:
            if not root.is_dir():
                continue
            searched.append(str(root))
            for current, dirs, files in os.walk(root, onerror=lambda error: errors.append(str(error))):
                dirs[:] = [item for item in dirs if item not in SKIP_CHECKPOINT_DIRS]
                for filename in files:
                    if Path(filename).suffix.lower() in CHECKPOINT_SUFFIXES:
                        found.append(str(Path(current, filename)))
        except OSError as error:
            errors.append(f"{root}: {error}")
    return {
        "searched_roots": searched, "candidate_count": len(found),
        "candidates": found[:50], "errors": errors,
    }


def audit_environment(workspace_root: Path, checkpoint_roots: list[str]) -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for package in ("numpy", "torch", "pytest", "Pillow", "navsim", "transformers"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    repositories = {
        name: _git_revision(workspace_root / name)
        for name in ("RePlanWorld", "WA-JEPA", "Drive-JEPA", "DriveDreamer-Policy", "SimWAM", "navsim")
    }
    return {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
        "nvidia_smi": _command_output([
            "nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"
        ]),
        "repository_revisions": repositories,
        "checkpoint_inventory": _checkpoint_inventory(workspace_root, checkpoint_roots),
    }


def run_audit(
    config_path: Path,
    workspace_root: Path,
    max_logs: int,
    frame_stride: int,
    selected_datasets: set[str] | None = None,
    checkpoint_dir: Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    if max_logs < 0 or frame_stride < 1:
        raise ValueError("max_logs must be >= 0 and frame_stride must be >= 1")
    with config_path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    cameras = [str(camera).upper() for camera in config["cameras"]]
    timestamp_unit = config.get("timestamp_unit", "microseconds")
    if timestamp_unit != "microseconds":
        raise ValueError("Only explicitly configured microsecond timestamps are supported")
    entries = [
        entry for entry in config["datasets"]
        if selected_datasets is None or entry["name"] in selected_datasets
    ]
    if not entries:
        raise ValueError("No configured datasets matched --dataset")
    datasets = [
        audit_dataset(
            entry["name"], workspace_root / entry["log_dir"],
            workspace_root / entry["sensor_dir"], cameras, timestamp_unit,
            max_logs, frame_stride,
            checkpoint_dir / f"{entry['name']}.jsonl" if checkpoint_dir else None,
            resume,
        )
        for entry in entries
    ]
    cross_dataset_log_overlap: list[dict[str, Any]] = []
    for left_index, left in enumerate(entries):
        left_dir = workspace_root / left["log_dir"]
        try:
            left_exists = left_dir.is_dir()
        except OSError:
            left_exists = False
        if not left_exists:
            continue
        left_names = {path.stem for path in left_dir.glob("*.pkl")}
        for right in entries[left_index + 1:]:
            right_dir = workspace_root / right["log_dir"]
            try:
                right_exists = right_dir.is_dir()
            except OSError:
                right_exists = False
            if not right_exists:
                continue
            right_names = {path.stem for path in right_dir.glob("*.pkl")}
            overlap = sorted(left_names & right_names)
            cross_dataset_log_overlap.append({
                "datasets": [left["name"], right["name"]],
                "shared_log_files": len(overlap),
                "examples": overlap[:10],
            })
    environment = audit_environment(workspace_root, config.get("checkpoint_search_roots", []))
    blockers: list[str] = []
    for dataset in datasets:
        if dataset["status"] != "ok":
            blockers.append(f"{dataset['name']}: {dataset['status']}")
            continue
        if dataset["coverage"]["mode"] != "full":
            blockers.append(f"{dataset['name']}: image availability checked only on a sample")
        if dataset["inventory"]["logs_without_sensor_directory"]:
            blockers.append(f"{dataset['name']}: some logs have no sensor directory")
        if any(counts.get("file_missing", 0) for counts in dataset["camera_totals_checked"].values()):
            blockers.append(f"{dataset['name']}: missing image files in checked frames")
    if not environment["nvidia_smi"]["available"]:
        blockers.append("GPU unavailable to nvidia-smi in this runtime")
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "workspace_root": str(workspace_root),
        "config_path": str(config_path),
        "timestamp_unit": timestamp_unit,
        "cameras": cameras,
        "datasets": datasets,
        "cross_dataset_log_overlap": cross_dataset_log_overlap,
        "environment": environment,
        "phase0_gate": {"decision": "NO_GO" if blockers else "REVIEW_REQUIRED", "blockers": blockers},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/audit.json"))
    parser.add_argument("--workspace-root", type=Path,
                        default=Path(__file__).resolve().parents[3])
    parser.add_argument("--max-logs-per-split", type=int, default=8,
                        help="0 scans all logs; positive N selects N evenly spaced logs")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--dataset", action="append", default=None,
                        help="Configured dataset name; repeat to select multiple")
    parser.add_argument("--checkpoint-dir", type=Path, default=None,
                        help="Save one resumable JSONL record per scanned log")
    parser.add_argument("--resume", action="store_true",
                        help="Reuse checkpoint records only when log and sensor directory metadata match")
    parser.add_argument("--output", type=Path, default=Path("reports/phase0/data_audit.json"))
    args = parser.parse_args()
    report = run_audit(args.config.resolve(), args.workspace_root.resolve(),
                       args.max_logs_per_split, args.frame_stride,
                       set(args.dataset) if args.dataset else None,
                       args.checkpoint_dir, args.resume)
    destination = args.output
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(destination)
    except OSError as error:
        destination = Path("/tmp/replanworld_data_audit_fallback.json")
        destination.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Workspace write failed ({error}); saved report to {destination}", file=sys.stderr)
    print(f"Wrote {destination}: {report['phase0_gate']['decision']}")
    for dataset in report["datasets"]:
        coverage = dataset.get("coverage", {})
        print(f"  {dataset['name']}: {coverage.get('mode', dataset['status'])}, "
              f"{coverage.get('scanned_log_count', 0)} scanned logs")
    for blocker in report["phase0_gate"]["blockers"]:
        print(f"  blocker: {blocker}")


if __name__ == "__main__":
    main()
