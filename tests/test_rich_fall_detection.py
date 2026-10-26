import math
from types import SimpleNamespace
import unittest

from fall_detection import (
    FallConfig, FallStateMachine, VelocityTracker,
    posture_features, full_body_features, is_down_posture,
    compute_fall_score, NORMAL, POSSIBLE_FALL, CONFIRMED_FALL,
)


def _make_landmarks(n=33, default_vis=1.0, overrides=None):
    """Build a list of fake landmarks for testing."""
    pts = [SimpleNamespace(x=0.5, y=0.5, visibility=default_vis) for _ in range(n)]
    for idx, x, y in (overrides or []):
        pts[idx].x, pts[idx].y = x, y
    return pts


class FallTests(unittest.TestCase):
    def test_sustained_horizontal_torso_can_reach_confirmation_gate(self):
        config = FallConfig()
        features = {"torso_ar": 1.4,"torso_axis_ar": 4.,"torso_y": .8,
                    "head_below_hips": False,"body_center_y": .8}
        score, signals = compute_fall_score(features,None,config)
        self.assertGreaterEqual(score,config.score_threshold)
        self.assertIn("horizontal_torso",signals)

    def test_wide_shoulders_and_high_pose_are_not_horizontal_falls(self):
        config = FallConfig()
        for features in (
            {"torso_ar":4.,"torso_axis_ar":.1,"torso_y":.8,"head_below_hips":False,"body_center_y":.8},
            {"torso_ar":4.,"torso_axis_ar":4.,"torso_y":.2,"head_below_hips":False,"body_center_y":.2}):
            score,_ = compute_fall_score(features,None,config)
            self.assertLess(score,config.score_threshold)

    def test_confirmation_and_recovery(self):
        fsm = FallStateMachine(FallConfig(confirm_seconds=2, recover_frames=2))
        self.assertEqual(fsm.update(True, 10), "POSSIBLE_FALL")
        self.assertEqual(fsm.update(True, 11.9), "POSSIBLE_FALL")
        self.assertEqual(fsm.update(True, 12), "CONFIRMED_FALL")
        self.assertEqual(fsm.update(False, 13), "CONFIRMED_FALL")
        self.assertEqual(fsm.update(False, 14), "NORMAL")

    def test_interruption_restarts_confirmation(self):
        fsm = FallStateMachine(FallConfig(confirm_seconds=3))
        fsm.update(True, 0)
        fsm.update(True, 2)
        fsm.update(False, 2.5)
        self.assertEqual(fsm.update(True, 3), "POSSIBLE_FALL")
        self.assertEqual(fsm.update(True, 5.9), "POSSIBLE_FALL")
        self.assertEqual(fsm.update(True, 6), "CONFIRMED_FALL")

    def test_time_validation_and_reset(self):
        fsm = FallStateMachine()
        fsm.update(True, 5)
        with self.assertRaises(ValueError):
            fsm.update(True, 4)
        with self.assertRaises(ValueError):
            fsm.update(True, math.nan)
        fsm.reset()
        self.assertEqual(fsm.update(True, 0), "POSSIBLE_FALL")

    def test_low_visibility_and_degenerate_pose(self):
        pts = _make_landmarks(overrides=[
            (11, .1, .65), (12, .8, .65), (23, .1, .75), (24, .8, .75),
        ])
        self.assertTrue(is_down_posture(pts, 480, 640))
        pts[11].visibility = 0.1
        self.assertIsNone(posture_features(pts, 480, 640))
        self.assertFalse(is_down_posture(pts, 480, 640))
        self.assertFalse(is_down_posture(None, 480, 640))

    def test_invalid_config(self):
        for values in ({"confirm_seconds": 0}, {"aspect_ratio_threshold": math.inf},
                       {"recover_frames": 1.5}, {"min_visibility": 2}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                FallConfig(**values)

    def test_velocity_window_validation(self):
        with self.assertRaises(ValueError):
            FallConfig(velocity_window=1)
        with self.assertRaises(ValueError):
            FallConfig(velocity_window=True)

    def test_full_body_features_head_below_hips(self):
        """Head below hips should be detected when nose.y > avg(hip.y) + margin."""
        pts = _make_landmarks(overrides=[
            (0, 0.5, 0.85),   # nose far below
            (11, 0.3, 0.4),   # L shoulder
            (12, 0.7, 0.4),   # R shoulder
            (23, 0.3, 0.6),   # L hip
            (24, 0.7, 0.6),   # R hip
        ])
        feats = full_body_features(pts, 480, 640)
        self.assertIsNotNone(feats)
        self.assertTrue(feats["head_below_hips"])

    def test_full_body_features_normal_standing(self):
        """Normal standing: head above hips."""
        pts = _make_landmarks(overrides=[
            (0, 0.5, 0.2),    # nose above
            (11, 0.4, 0.35),  # L shoulder
            (12, 0.6, 0.35),  # R shoulder
            (23, 0.4, 0.55),  # L hip
            (24, 0.6, 0.55),  # R hip
        ])
        feats = full_body_features(pts, 480, 640)
        self.assertIsNotNone(feats)
        self.assertFalse(feats["head_below_hips"])

    def test_velocity_tracker_detects_sudden_drop(self):
        vt = VelocityTracker(window=3)
        # Simulate body moving down rapidly.
        self.assertIsNone(vt.update(0.3, 0.0))
        v = vt.update(0.5, 0.1)   # 0.2 / 0.1 = 2.0 units/sec
        self.assertIsNotNone(v)
        self.assertGreater(v, 1.0)

    def test_velocity_tracker_no_movement(self):
        vt = VelocityTracker(window=3)
        vt.update(0.5, 0.0)
        v = vt.update(0.5, 1.0)
        self.assertAlmostEqual(v, 0.0, places=3)

    def test_compute_fall_score_all_signals(self):
        """All three signals active should produce a high score."""
        config = FallConfig()
        features = {"torso_ar": 2.0, "torso_y": 0.7, "head_below_hips": True, "body_center_y": 0.7}
        score, breakdown = compute_fall_score(features, velocity=0.5, config=config)
        self.assertGreaterEqual(score, config.score_threshold)
        self.assertIn("head_below_hips", breakdown)
        self.assertIn("velocity", breakdown)

    def test_compute_fall_score_normal_posture(self):
        """No signals fired should give zero score."""
        config = FallConfig()
        features = {"torso_ar": 0.5, "torso_y": 0.3, "head_below_hips": False, "body_center_y": 0.3}
        score, _ = compute_fall_score(features, velocity=0.0, config=config)
        self.assertEqual(score, 0.0)

    def test_update_rich_confirms_fall(self):
        """update_rich should detect fall from landmarks with head below hips + wide torso."""
        config = FallConfig(confirm_seconds=0.5)
        fsm = FallStateMachine(config)
        # Simulate a fallen pose: wide torso, head below hips.
        pts = _make_landmarks(overrides=[
            (0, 0.5, 0.9),    # nose very low
            (11, 0.1, 0.65),  # L shoulder wide
            (12, 0.9, 0.65),  # R shoulder wide
            (23, 0.1, 0.75),  # L hip
            (24, 0.9, 0.75),  # R hip
            (25, 0.3, 0.8),   # L knee
            (26, 0.7, 0.8),   # R knee
            (27, 0.3, 0.85),  # L ankle
            (28, 0.7, 0.85),  # R ankle
        ])
        state = fsm.update_rich(pts, 480, 640, timestamp=0.0)
        self.assertIn(state, (POSSIBLE_FALL, CONFIRMED_FALL))
        # Continue feeding for enough time to confirm.
        for t in [0.2, 0.4, 0.6]:
            state = fsm.update_rich(pts, 480, 640, timestamp=t)
        self.assertEqual(state, CONFIRMED_FALL)

    def test_config_load_ignores_unknown_keys(self):
        """Loading a config with extra keys shouldn't crash (forward compatibility)."""
        import tempfile, json
        data = {"confirm_seconds": 2.0, "future_option": 42}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            config = FallConfig.load(f.name)
        self.assertEqual(config.confirm_seconds, 2.0)
