import unittest
from video_stream import frames


class Capture:
    def __init__(self, values=(), opened=True):
        self.values = iter(values)
        self.opened = opened
        self.released = False

    def isOpened(self):
        return self.opened

    def read(self):
        return next(self.values, (False, None))

    def release(self):
        self.released = True


class StreamTests(unittest.TestCase):
    def test_failed_reconnect_then_success(self):
        captures = [Capture([(True, "first")]), Capture(opened=False), Capture([(True, "second")])]
        available = iter(captures)
        stream = frames("rtsp://example", capture_factory=lambda _: next(available), sleep=lambda _: None)
        self.assertEqual(next(stream), ("first", False))
        self.assertEqual(next(stream), ("second", True))
        stream.close()
        self.assertTrue(all(c.released for c in captures))

    def test_initial_failures_are_bounded_and_released(self):
        captures = []
        def factory(_):
            cap = Capture(opened=False)
            captures.append(cap)
            return cap
        with self.assertRaises(RuntimeError):
            list(frames("rtsp://example", 2, capture_factory=factory, sleep=lambda _: None))
        self.assertEqual(len(captures), 3)
        self.assertTrue(all(c.released for c in captures))

    def test_file_ends_without_reconnection(self):
        cap = Capture([(True, "frame")])
        self.assertEqual(list(frames("clip.mp4", capture_factory=lambda _: cap)), [("frame", False)])
        self.assertTrue(cap.released)

    def test_unavailable_local_camera_reports_failure(self):
        cap = Capture(opened=False)
        with self.assertRaises(RuntimeError):
            list(frames(0, capture_factory=lambda _: cap))
        self.assertTrue(cap.released)
