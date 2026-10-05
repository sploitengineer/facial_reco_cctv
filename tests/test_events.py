import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import phase5_production


class EventTests(unittest.TestCase):
    def test_csv_quotes_names_and_snapshot_failure_is_visible(self):
        cv2 = Mock()
        cv2.imwrite.return_value = False
        with tempfile.TemporaryDirectory() as directory, patch.object(
                phase5_production, "EVENT_LOG_DIR", Path(directory)), patch.dict("sys.modules", {"cv2": cv2}):
            with self.assertLogs("SentryCare", level="WARNING") as logs:
                self.assertEqual(phase5_production.send_alert('Resident, "A"', "FALL", frame=object()), "console")
            self.assertTrue(any("Could not save" in entry for entry in logs.output))
            path = next(Path(directory).glob("events_*.csv"))
            with path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["person"], 'Resident, "A"')
            self.assertEqual(rows[0]["alert_sent"], "console")

    def test_sms_failure_is_recorded(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
                phase5_production, "EVENT_LOG_DIR", Path(directory)), patch.dict(
                "sys.modules", {"cv2": Mock(), "twilio": None, "twilio.rest": None}):
            with self.assertLogs("SentryCare", level="WARNING"):
                result = phase5_production.send_alert("Resident", "FALL", phone="test-number")
            self.assertEqual(result, "sms_unavailable")
            path = next(Path(directory).glob("events_*.csv"))
            with path.open(newline="", encoding="utf-8") as stream:
                self.assertEqual(next(csv.DictReader(stream))["alert_sent"], "sms_unavailable")
