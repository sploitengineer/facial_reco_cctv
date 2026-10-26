import unittest

from sentrycare.face_identity import FaceTracker


class FaceTrackerTests(unittest.TestCase):
    def test_basic_tracking(self):
        tracker = FaceTracker(stale_timeout=5.0, iou_threshold=0.3)
        # First frame: two faces.
        faces1 = [("alice", (100, 100, 50, 50)), ("unknown", (300, 100, 50, 50))]
        result = tracker.update(faces1, timestamp=1.0)
        self.assertEqual(len(result), 2)

    def test_explicit_unknown_clears_previous_identity(self):
        tracker = FaceTracker(stale_timeout=5.0, iou_threshold=0.3)
        # Frame 1: alice detected.
        tracker.update([("alice", (100, 100, 50, 50))], timestamp=1.0)
        # Frame 2: explicit unknown must not inherit a previous identity.
        result = tracker.update([("unknown", (105, 102, 50, 50))], timestamp=2.0)
        self.assertEqual(result[0][0], "unknown")

    def test_stale_tracks_removed(self):
        tracker = FaceTracker(stale_timeout=1.0, iou_threshold=0.3)
        tracker.update([("bob", (100, 100, 50, 50))], timestamp=1.0)
        # 3 seconds later with no detections.
        result = tracker.update([], timestamp=4.0)
        self.assertEqual(len(result), 0)

    def test_new_face_gets_new_track(self):
        tracker = FaceTracker(stale_timeout=5.0, iou_threshold=0.3)
        tracker.update([("alice", (100, 100, 50, 50))], timestamp=1.0)
        # New face far from alice.
        result = tracker.update([("alice", (100, 100, 50, 50)), ("bob", (500, 100, 50, 50))], timestamp=2.0)
        names = sorted([n for n, _ in result])
        self.assertIn("alice", names)
        self.assertIn("bob", names)

    def test_reset_clears_all_tracks(self):
        tracker = FaceTracker()
        tracker.update([("alice", (100, 100, 50, 50))], timestamp=1.0)
        tracker.reset()
        result = tracker.update([], timestamp=2.0)
        self.assertEqual(len(result), 0)

    def test_known_name_overrides_unknown(self):
        """When a face was unknown and gets identified, the name should update."""
        tracker = FaceTracker(stale_timeout=5.0, iou_threshold=0.3)
        tracker.update([("unknown", (100, 100, 50, 50))], timestamp=1.0)
        result = tracker.update([("charlie", (102, 101, 50, 50))], timestamp=2.0)
        self.assertEqual(result[0][0], "charlie")
