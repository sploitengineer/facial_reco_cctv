import io
import tempfile
import time
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from sentrycare.registry import Registry
from sentrycare.studio import Studio
from sentrycare.vision import match_face, normalized
from sentrycare.web import create_app


def vector(index):
    result = np.zeros(128, dtype=np.float32)
    result[index] = 1
    return result


class FakeEngine:
    def __init__(self):
        self.faces = 1
        self.warnings = []
        self.index = 0
    def initialize(self):
        pass
    def detect(self, frame):
        return [{"aligned": np.full((112, 112, 3), 100, np.uint8), "bbox": [5 + 110*i, 5, 100, 100],
                 "quality": {"sharpness": 100, "brightness": 100, "face_pixels": 100},
                 "warnings": self.warnings} for i in range(self.faces)]
    def embed(self, _):
        return vector(self.index)


class MatcherTests(unittest.TestCase):
    def test_matches_cosine_and_rejects_unknown(self):
        gallery = [{"id": "a", "name": "Alice", "vectors": [vector(0)] * 3}]
        self.assertEqual(match_face(vector(0), gallery)["name"], "Alice")
        self.assertEqual(match_face(vector(1), gallery)["name"], "Unknown")
    def test_similar_identities_are_rejected(self):
        gallery = [{"id": "a", "name": "Alice", "vectors": [vector(0)]},
                   {"id": "b", "name": "Bob", "vectors": [normalized(vector(0) + .1*vector(1))]}]
        self.assertEqual(match_face(vector(0), gallery)["reason"], "Ambiguous match")
    def test_invalid_embeddings_rejected(self):
        for value in (np.zeros(128), np.zeros(127), np.full(128, np.nan)):
            with self.assertRaises(ValueError):
                normalized(value)


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.registry = Registry(self.temp.name)
        self.person = self.registry.create_person("Alice")
    def add(self, source=None):
        return self.registry.save_sample(self.person["id"], np.full((112,112,3),100,np.uint8), "front", {}, source)
    def test_duplicate_names_are_case_insensitive(self):
        with self.assertRaises(ValueError):
            self.registry.create_person(" ALICE ")
    def test_identity_has_no_path_privileges(self):
        person = self.registry.create_person("../people / Alice")
        self.assertEqual(len(person["id"]), 32)
        self.assertFalse(any(p.is_dir() for p in self.registry.images.iterdir()))
    def test_import_is_idempotent(self):
        self.assertIsNotNone(self.add("imagehash"))
        self.assertIsNone(self.add("imagehash"))
        self.assertEqual(self.registry.person(self.person["id"])["sample_count"], 1)
    def test_training_requires_three_samples_and_persists(self):
        ids = [self.add() for _ in range(3)]
        self.assertEqual(self.registry.gallery(), [])
        self.registry.save_embeddings([(sample,vector(0)) for sample in ids])
        self.assertEqual(self.registry.person(self.person["id"])["status"], "ready")
        reopened = Registry(self.temp.name)
        self.assertEqual(reopened.gallery()[0]["name"], "Alice")
        self.registry.delete_sample(ids[0])
        self.assertEqual(self.registry.gallery(), [])
    def test_person_deletion_removes_features_and_images(self):
        ids = [self.add() for _ in range(3)]
        self.registry.save_embeddings([(sample,vector(0)) for sample in ids])
        self.assertTrue(self.registry.delete_person(self.person["id"]))
        self.assertEqual(self.registry.people(), [])
        self.assertEqual(list(self.registry.images.iterdir()), [])
        self.assertEqual(self.registry.gallery(), [])
    def test_new_photo_marks_profile_for_update(self):
        ids = [self.add() for _ in range(3)]
        self.registry.save_embeddings([(sample,vector(0)) for sample in ids])
        self.add()
        self.assertEqual(self.registry.person(self.person["id"])["status"], "update_needed")


class WebTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.engine = FakeEngine()
        self.studio = Studio(self.temp.name, engine=self.engine, autostart=False)
        self.studio.status = "ready"
        self.client = create_app(self.studio).test_client()
        self.token = self.client.get("/api/state").json["token"]
        self.headers = {"X-SentryCare-Token": self.token}
        self.frame_headers = {**self.headers, "X-Session-ID": "testsession123"}
        self.image = cv2.imencode(".jpg", np.full((240,320,3),100,np.uint8))[1].tobytes()
        self.person = self.client.post("/api/people", json={"name":"Alice"}, headers=self.headers).json
    def sample(self):
        return self.client.post(f"/api/people/{self.person['id']}/samples?angle=front",
                                data=self.image,content_type="image/jpeg",headers=self.headers)
    def test_html_and_csrf(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.client.post("/api/people",json={"name":"Bad"}).status_code,403)
        self.assertEqual(self.client.post("/api/people",json={"name":"Bad"},headers={"X-SentryCare-Token":"bad"}).status_code,403)
    def test_nonlocal_host_rejected(self):
        self.assertEqual(self.client.get("/api/state",headers={"Host":"attacker.example"}).status_code,403)
    def test_quality_and_group_photos_rejected(self):
        self.engine.faces = 2
        self.assertEqual(self.sample().status_code,400)
        self.engine.faces = 0
        self.assertEqual(self.sample().status_code,400)
        self.engine.faces = 1
        self.engine.warnings = ["Photo is blurry"]
        response = self.sample()
        self.assertEqual(response.status_code,400)
        self.assertIn("blurry",response.json["error"])
        self.assertEqual(self.studio.registry.person(self.person["id"])["sample_count"],0)
    def test_bad_image_and_bad_angle(self):
        self.assertEqual(self.client.post(f"/api/people/{self.person['id']}/samples",data=b"not-image",headers=self.headers).status_code,400)
        self.assertEqual(self.client.post(f"/api/people/{self.person['id']}/samples?angle=bad",data=self.image,headers=self.headers).status_code,400)
    def test_enroll_train_match_delete(self):
        for _ in range(3):
            self.assertEqual(self.sample().status_code,201)
        response = self.client.post("/api/train",json={"person_id":self.person["id"]},headers=self.headers)
        self.assertEqual(response.status_code,202)
        job_id = response.json["id"]
        for _ in range(100):
            job = self.client.get(f"/api/jobs/{job_id}").json
            if job["status"] != "running":
                break
            time.sleep(.01)
        self.assertEqual(job["status"],"complete")
        self.assertEqual(self.studio.registry.people()[0]["status"],"ready")
        self.studio.refresh_gallery()
        self.assertEqual(match_face(vector(0),self.studio.gallery)["name"],"Alice")
        sample = self.studio.registry.people()[0]["samples"][0]
        with self.client.get(sample["image_url"]) as image_response:
            self.assertEqual(image_response.status_code,200)
        self.assertEqual(self.client.delete(f"/api/people/{self.person['id']}",headers=self.headers).status_code,200)
        self.studio.refresh_gallery()
        self.assertEqual(match_face(vector(0),self.studio.gallery)["name"],"Unknown")
    def test_no_body_needed_for_enrollment_analysis(self):
        result = self.client.post("/api/frame?mode=enroll",data=self.image,content_type="image/jpeg",headers=self.frame_headers)
        self.assertEqual(result.status_code,200)
        self.assertEqual(len(result.json["faces"]),1)
        self.assertFalse(result.json["fall"]["pose_usable"])
    def test_busy_frame_drops_instead_of_queuing(self):
        self.studio.inference_lock.acquire()
        try:
            response = self.client.post("/api/frame?mode=enroll",data=self.image,headers=self.frame_headers)
            self.assertEqual(response.status_code,202)
        finally:
            self.studio.inference_lock.release()
    def test_invalid_session_rejected(self):
        self.assertEqual(self.client.post("/api/frame",data=self.image,headers=self.headers).status_code,400)

    def test_live_identity_consensus_unknown_and_gallery_invalidation(self):
        for _ in range(3):
            self.sample()
        rows = self.studio.registry.sample_rows()
        self.studio.registry.save_embeddings([(r["id"],vector(0)) for r in rows])
        class Pose:
            def process(self, _):
                return SimpleNamespace(pose_landmarks=None)
        self.studio.mp = SimpleNamespace(solutions=SimpleNamespace(pose=SimpleNamespace(Pose=lambda **kw:Pose())))
        frame = cv2.imdecode(np.frombuffer(self.image,np.uint8),cv2.IMREAD_COLOR)
        with patch("sentrycare.studio.time.monotonic",return_value=10):
            first = self.studio.analyze(frame,"testsession123")
        self.assertEqual(first["faces"][0]["name"],"Checking")
        with patch("sentrycare.studio.time.monotonic",return_value=10.8):
            second = self.studio.analyze(frame,"testsession123")
        self.assertEqual(second["faces"][0]["name"],"Alice")
        self.engine.index = 1
        with patch("sentrycare.studio.time.monotonic",return_value=11.6):
            unknown = self.studio.analyze(frame,"testsession123")
        self.assertEqual(unknown["faces"][0]["name"],"Unknown")
        self.studio.registry.delete_person(self.person["id"])
        with patch("sentrycare.studio.time.monotonic",return_value=11.7):
            deleted = self.studio.analyze(frame,"testsession123")
        self.assertEqual(deleted["faces"][0]["name"],"Unknown")
