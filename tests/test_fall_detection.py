import math
from types import SimpleNamespace
import unittest

from fall_detection import FallConfig, FallStateMachine, posture_features, is_down_posture


class FallTests(unittest.TestCase):
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
        pts = [SimpleNamespace(x=0.2, y=0.7, visibility=1) for _ in range(33)]
        for i, x, y in ((11, .1, .65), (12, .8, .65), (23, .1, .75), (24, .8, .75)):
            pts[i].x, pts[i].y = x, y
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
