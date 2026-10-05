from pathlib import Path
import tempfile
import unittest

from sentrycare.recordings import sequence_entries


class SequenceTests(unittest.TestCase):
    def test_numeric_order_and_recorded_timestamps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("frame10.png", "frame2.png", "frame1.png"):
                (root / name).touch()
            timing = root / "timing.csv"
            timing.write_text("1,0,1\n2,33,1\n10,315,1\n")
            entries = sequence_entries(root, timing)
            self.assertEqual([p.name for p, _ in entries], ["frame1.png", "frame2.png", "frame10.png"])
            self.assertEqual([t for _, t in entries], [0, .033, .315])

    def test_missing_timing_and_duplicate_frames_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "frame1.png").touch()
            timing = root / "timing.csv"
            timing.write_text("2,0,1\n")
            with self.assertRaises(ValueError):
                sequence_entries(root, timing)
            timing.write_text("1,0,1\n")
            (root / "other1.png").touch()
            with self.assertRaises(ValueError):
                sequence_entries(root, timing)
