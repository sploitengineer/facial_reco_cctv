import tempfile
import unittest
from unittest.mock import Mock

from face_identity import identify_face


class Frame:
    shape = (100, 100, 3)
    def __getitem__(self, _):
        return self


class IdentityTests(unittest.TestCase):
    def test_empty_enrollment_still_detects_unknown_with_integer_box(self):
        backend = Mock()
        backend.extract_faces.return_value = [{"facial_area": {"x": 10.0, "y": 20.0, "w": 30.0, "h": 40.0}}]
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(identify_face(Frame(), directory, backend=backend), ("unknown", (10, 20, 30, 40)))
        backend.find.assert_not_called()
        self.assertTrue(backend.extract_faces.call_args.kwargs["enforce_detection"])

    def test_no_face_is_not_unknown(self):
        backend = Mock()
        backend.extract_faces.side_effect = ValueError("Face could not be detected")
        self.assertEqual(identify_face(Frame(), backend=backend), (None, None))
        backend.find.assert_not_called()
