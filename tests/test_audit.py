import errno
import pickle
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from replan_world.data.audit import audit_dataset, run_audit, safe_sensor_path, scan_log, select_log_files


class AuditTests(unittest.TestCase):
    def test_select_log_files_includes_edges(self):
        files = [Path(f"{index:03}.pkl") for index in range(9)]
        self.assertEqual([p.name for p in select_log_files(files, 3)],
                         ["000.pkl", "004.pkl", "008.pkl"])
        self.assertEqual(select_log_files(files, 0), files)

    def test_sensor_path_rejects_escape(self):
        root = Path("/sensors")
        self.assertEqual(safe_sensor_path(root, "log/CAM_F0/a.jpg"),
                         root / "log/CAM_F0/a.jpg")
        self.assertIsNone(safe_sensor_path(root, "../private"))
        self.assertIsNone(safe_sensor_path(root, "/absolute/file.jpg"))

    def test_dataset_filter_rejects_unknown_name(self):
        config = Path(__file__).resolve().parents[1] / "configs" / "audit.json"
        with self.assertRaisesRegex(ValueError, "No configured datasets"):
            run_audit(config, Path("/unused"), 1, 1, {"unknown"})

    def test_scan_counts_missing_images_and_timestamps(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log_dir = root / "logs"
            sensor_dir = root / "sensors"
            log_dir.mkdir()
            image_dir = sensor_dir / "sample" / "CAM_F0"
            image_dir.mkdir(parents=True)
            (image_dir / "a.jpg").write_bytes(b"image")
            frames = [
                {
                    "timestamp": 1_000_000 + i * 500_000,
                    "ego2global_translation": [0, 0, 0],
                    "ego2global_rotation": [1, 0, 0, 0],
                    "ego_dynamic_state": [0, 0, 0, 0],
                    "driving_command": [0, 1, 0, 0],
                    "cams": {"CAM_F0": {"data_path": f"sample/CAM_F0/{name}.jpg"}},
                }
                for i, name in enumerate(("a", "b"))
            ]
            log_path = log_dir / "sample.pkl"
            with log_path.open("wb") as stream:
                pickle.dump(frames, stream)

            result = scan_log(log_path, sensor_dir, {"sample"}, ["CAM_F0", "CAM_L0"],
                              "microseconds")
            self.assertEqual(result["timestamps"]["median_interval_seconds"], 0.5)
            self.assertEqual(result["pose_valid_frames"], 2)
            self.assertEqual(result["camera_counts"]["CAM_F0"]["file_found"], 1)
            self.assertEqual(result["camera_counts"]["CAM_F0"]["file_missing"], 1)
            self.assertEqual(result["camera_counts"]["CAM_L0"]["metadata_missing"], 2)

            with mock.patch("replan_world.data.audit.os.scandir",
                            side_effect=OSError(errno.ENOTCONN, "mount disconnected")):
                with self.assertRaises(OSError):
                    scan_log(log_path, sensor_dir, {"sample"}, ["CAM_F0"],
                             "microseconds")

            dataset = audit_dataset("fake", log_dir, sensor_dir, ["CAM_F0", "CAM_L0"],
                                    "microseconds", 0, 1)
            self.assertEqual(dataset["coverage"]["mode"], "full")
            self.assertEqual(dataset["inventory"]["logs_with_sensor_directory"], 1)
            self.assertEqual(dataset["camera_totals_checked"]["CAM_F0"]["file_missing"], 1)

            checkpoint = root / "progress.jsonl"
            first = audit_dataset("fake", log_dir, sensor_dir, ["CAM_F0"],
                                  "microseconds", 0, 1, checkpoint, False)
            cached = audit_dataset("fake", log_dir, sensor_dir, ["CAM_F0"],
                                   "microseconds", 0, 1, checkpoint, True)
            self.assertEqual(first["coverage"]["resumed_log_count"], 0)
            self.assertEqual(cached["coverage"]["resumed_log_count"], 1)
            (image_dir / "b.jpg").write_bytes(b"image")
            refreshed = audit_dataset("fake", log_dir, sensor_dir, ["CAM_F0"],
                                      "microseconds", 0, 1, checkpoint, True)
            self.assertEqual(refreshed["coverage"]["resumed_log_count"], 0)
            self.assertEqual(refreshed["camera_totals_checked"]["CAM_F0"]["file_found"], 2)


if __name__ == "__main__":
    unittest.main()
