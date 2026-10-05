import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fall_detection import FallConfig
from phase6_evaluation import evaluate, load_features, tune, main


def clip(name, label, split, aspect):
    return {"path": name, "label": label, "split": split,
            "frames": [{"timestamp": i * 0.5, "aspect_ratio": aspect, "torso_y": .7} for i in range(9)]}


class EvaluationTests(unittest.TestCase):
    def test_metrics_use_clip_time(self):
        metrics = evaluate([clip("fall", "fall", "train", 2), clip("adl", "adl", "train", .5)], FallConfig())
        self.assertEqual((metrics["tp"], metrics["tn"], metrics["f1"]), (1, 1, 1))
        self.assertEqual(metrics["predictions"][0]["first_alert_seconds"], 3)

    def test_grid_selection_never_uses_validation(self):
        training = [clip("f", "fall", "train", 1.4), clip("a", "adl", "train", 1.2)]
        config, report = tune(training + [clip("v", "fall", "validation", 1.2)], [1.1, 1.3], [.55], [1])
        self.assertEqual(config.aspect_ratio_threshold, 1.3)
        self.assertEqual(report["train"]["f1"], 1)
        self.assertEqual(report["validation"]["fn"], 1)

    def test_no_pose_does_not_trigger(self):
        data = clip("f", "fall", "train", None)
        for frame in data["frames"]:
            frame["torso_y"] = None
        self.assertEqual(evaluate([data], FallConfig())["fn"], 1)

    def test_requires_both_training_classes(self):
        with self.assertRaises(ValueError):
            tune([clip("f", "fall", "train", 2)], [1.3], [.55], [3])

    def test_zero_training_score_is_flagged(self):
        _, report = tune([clip("f", "fall", "train", .5), clip("a", "adl", "train", .5)],
                         [1.3], [.55], [3])
        self.assertFalse(report["training_detected_falls"])

    def test_duplicate_and_invalid_timestamps_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.json"
            data = clip("f", "fall", "train", 2)
            for clips in ([data, data], [dict(data, frames=[data["frames"][0]] * 2)]):
                path.write_text(json.dumps({"schema_version": 1, "min_visibility": .5, "clips": clips}))
                with self.assertRaises(ValueError):
                    load_features(path)

    def test_cli_exports_config_and_held_out_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features = root / "features.json"
            features.write_text(json.dumps({"schema_version": 1, "min_visibility": .5,
                "clips": [clip("f", "fall", "train", 2), clip("a", "adl", "train", .5),
                          clip("v", "fall", "validation", 2)]}))
            config_path, report_path = root / "config.json", root / "report.json"
            with patch("sys.argv", ["phase6_evaluation.py", "tune", str(features),
                         "--config-output", str(config_path), "--report", str(report_path)]):
                main()
            config = FallConfig.load(config_path)
            self.assertGreater(config.confirm_seconds, 0)
            report = json.loads(report_path.read_text())
            self.assertEqual(report["candidates_evaluated"], 27)
            self.assertEqual(report["validation"]["tp"], 1)
            self.assertIsNone(report["test"])
