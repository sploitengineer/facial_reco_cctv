import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from sentrycare.cameras import Worker, load_config, stalled, stop_worker, worker_command, supervise


class CameraTests(unittest.TestCase):
    def load(self, config):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config))
            return load_config(path)

    def test_env_source_and_disabled_camera(self):
        with patch.dict(os.environ, {"TEST_RTSP": "rtsp://example"}):
            result = self.load({"cameras": [{"id": "one", "source_env": "TEST_RTSP"},
                                             {"id": "two", "enabled": False}]})
        self.assertEqual(len(result["cameras"]), 1)
        self.assertEqual(result["cameras"][0]["source"], "rtsp://example")

    def test_invalid_ids_sources_and_limits(self):
        for config in ({"cameras": [{"id": "../one", "source": 0}]},
                       {"cameras": [{"id": "a", "source": 0}, {"id": "b", "source": 0}]},
                       {"cameras": [{"id": "a", "source": 0}], "restart_limit": -1},
                       {"cameras": [{"id": "a", "source": True}]},
                       {"cameras": [{"id": "a", "source": 0}], "stall_timeout": float("inf")}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                self.load(config)

    def test_worker_command_has_isolated_camera_and_no_gui(self):
        command = worker_command({"id": "one", "source": "0", "face_id": False}, Path("progress.json"))
        self.assertIn("--headless", command)
        self.assertIn("--no-face-id", command)
        self.assertIn("one", command)

    def test_watchdog_ignores_heartbeat_from_previous_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "heartbeat.json"
            path.write_text("{}")
            os.utime(path, (10, 10))
            worker = Worker({}, path, started_at=20)
            self.assertFalse(stalled(worker, 25, 10, 2))
            self.assertTrue(stalled(worker, 31, 10, 2))
            os.utime(path, (30, 30))
            self.assertFalse(stalled(worker, 31, 10, 2))
            self.assertTrue(stalled(worker, 33, 10, 2))

    def test_shutdown_terminates_worker_and_closes_logs(self):
        worker = Worker({}, Path("unused"), process=Mock(), log_handle=Mock())
        worker.process.poll.return_value = None
        handle = worker.log_handle
        stop_worker(worker)
        worker.process.terminate.assert_called_once()
        handle.close.assert_called_once()

    def test_bounded_restarts_and_final_status(self):
        config = self.load({"restart_limit": 1, "restart_delay": .01,
                            "cameras": [{"id": "one", "source": 0, "face_id": False}]})
        process = Mock()
        process.poll.return_value = 1
        with tempfile.TemporaryDirectory() as directory, patch("sentrycare.cameras.LOGS", Path(directory)), \
                patch("sentrycare.cameras.ensure_data_dirs"), patch("sentrycare.cameras.subprocess.Popen", return_value=process) as launch:
            self.assertEqual(supervise(config), 1)
            self.assertEqual(launch.call_count, 2)
            status = json.loads((Path(directory) / "cameras_status.json").read_text())
            self.assertEqual(status["cameras"][0]["status"], "exhausted")

    def test_failed_camera_does_not_stop_healthy_camera(self):
        config = self.load({"restart_limit": 0, "cameras": [
            {"id": "bad", "source": 0}, {"id": "good", "source": 1}]})
        failed, healthy = Mock(), Mock()
        failed.poll.return_value = 1
        healthy.poll.return_value = None
        with tempfile.TemporaryDirectory() as directory, patch("sentrycare.cameras.LOGS", Path(directory)), \
                patch("sentrycare.cameras.ensure_data_dirs"), \
                patch("sentrycare.cameras.subprocess.Popen", side_effect=[failed, healthy]) as launch, \
                patch("sentrycare.cameras.stalled", return_value=False), \
                patch("sentrycare.cameras.time.sleep", side_effect=KeyboardInterrupt):
            self.assertEqual(supervise(config), 0)
            self.assertEqual(launch.call_count, 2)
            status = json.loads((Path(directory) / "cameras_status.json").read_text())
            self.assertEqual([c["status"] for c in status["cameras"]], ["exhausted", "stopped"])
            healthy.terminate.assert_called_once()
