"""Shared enrollment jobs and bounded, latest-frame recognition/pose processing."""

from dataclasses import replace
import hashlib
import json
import logging
from pathlib import Path
import threading
import time
from uuid import uuid4
import zipfile

import cv2
import numpy as np

from .fall_detection import FallConfig, FallStateMachine, CONFIRMED_FALL, studio_fall_config
from .face_identity import FaceTracker
from .registry import Registry
from .vision import FaceEngine, match_face

logger = logging.getLogger("SentryCare")


def decode_image(data):
    if not data or len(data) > 8 * 1024 * 1024:
        raise ValueError("Choose a JPEG or PNG smaller than 8 MB")
    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        raise ValueError("This file is not a readable image")
    height, width = frame.shape[:2]
    if width * height > 20_000_000:
        raise ValueError("Image is too large; choose one below 20 megapixels")
    if max(width, height) > 1600:
        scale = 1600 / max(width, height)
        frame = cv2.resize(frame, (round(width * scale), round(height * scale)))
    return frame


class Studio:
    def __init__(self, root, engine=None, autostart=True, project_root=None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.registry = Registry(self.root)
        self.engine = engine or FaceEngine()
        self.project_root = Path(project_root or Path(__file__).resolve().parent.parent)
        self.status = "starting"
        self.error = None
        self.inference_lock = threading.Lock()
        self.job_lock = threading.Lock()
        self.jobs = {}
        self.sessions = {}
        self.gallery_version = -1
        self.gallery = []
        self.settings = {"similarity_threshold": .45, "confirm_seconds": 1.2}
        self.mp = None
        self.active_job = None
        if autostart:
            threading.Thread(target=self.initialize, daemon=True, name="studio-models").start()

    def initialize(self):
        try:
            self.engine.initialize()
            import mediapipe as mp
            self.mp = mp
            self.status = "ready"
        except Exception as exc:
            logger.exception("Studio initialization failed")
            self.status, self.error = "error", str(exc)

    def require_ready(self):
        if self.status != "ready":
            raise RuntimeError(self.error or "Models are loading. Please wait for Ready.")

    def refresh_gallery(self):
        if self.gallery_version != self.registry.version:
            self.gallery = self.registry.gallery()
            self.gallery_version = self.registry.version
            for session in self.sessions.values():
                session["tracks"] = []

    def enroll(self, person_id, frame, angle="front", source_key=None):
        self.require_ready()
        faces = self.engine.detect(frame)
        if len(faces) != 1:
            raise ValueError("Keep exactly one face in the picture" if faces else "No face found. Face the camera and improve the lighting.")
        face = faces[0]
        if face["warnings"]:
            raise ValueError(". ".join(face["warnings"]))
        return self.registry.save_sample(person_id, face["aligned"], angle, face["quality"], source_key)

    def start_job(self, kind, person_id=None):
        self.require_ready()
        if not self.job_lock.acquire(blocking=False):
            raise ValueError("A training/import job is already running")
        if person_id and self.registry.person(person_id) is None:
            self.job_lock.release()
            raise ValueError("Profile does not exist")
        job_id = uuid4().hex
        job = {"id": job_id, "kind": kind, "status": "running", "done": 0, "total": 0,
               "accepted": 0, "errors": [], "person_id": person_id}
        self.jobs[job_id] = job
        self.active_job = job_id
        def run():
            try:
                if kind == "train":
                    self.train(job, person_id)
                elif kind == "import":
                    self.import_existing(job)
                job["status"] = "complete"
            except Exception as exc:
                logger.exception("Enrollment job failed")
                job["status"] = "error"
                job["errors"].append(str(exc))
            finally:
                self.active_job = None
                self.job_lock.release()
        threading.Thread(target=run, daemon=True, name=f"studio-{kind}").start()
        return job

    def train(self, job, person_id=None):
        rows = self.registry.sample_rows(person_id)
        if person_id and len(rows) < 3:
            raise ValueError("Save at least 3 clear photos before training")
        job["total"] = len(rows)
        embeddings = []
        for row in rows:
            image = cv2.imread(str(self.registry.images / f"{row['id']}.jpg"))
            if image is None:
                job["errors"].append(f"Missing photo for {row['name']}")
            else:
                embeddings.append((row["id"], self.engine.embed(image)))
                job["accepted"] += 1
            job["done"] += 1
        # Publish all embeddings in one transaction; readers keep the last ready gallery until this point.
        self.registry.save_embeddings(embeddings)

    def import_existing(self, job):
        candidates = []
        known = self.project_root / "data/known_faces"
        for image in known.glob("*/*"):
            if image.is_file() and image.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                candidates.append((image.parent.name, image.name, lambda p=image: p.read_bytes()))
        archive_path = self.project_root / "facial_reco_cctv.zip"
        archive = zipfile.ZipFile(archive_path) if archive_path.is_file() else None
        try:
            if archive:
                prefix = "facial_reco_cctv/data/known_faces/"
                for info in archive.infolist():
                    relative = info.filename.removeprefix(prefix).split("/")
                    if (info.filename.startswith(prefix) and len(relative) == 2 and not info.is_dir()
                            and Path(relative[1]).suffix.lower() in (".jpg", ".jpeg", ".png")
                            and info.file_size <= 8 * 1024 * 1024):
                        candidates.append((relative[0], relative[1], lambda n=info.filename: archive.read(n)))
            job["total"] = len(candidates)
            people = {p["name"].casefold(): p["id"] for p in self.registry.people()}
            for name, filename, read in candidates:
                try:
                    person_id = people.get(name.casefold())
                    if not person_id:
                        person_id = self.registry.create_person(name)["id"]
                        people[name.casefold()] = person_id
                    data = read()
                    source_key = hashlib.sha256(name.casefold().encode() + data).hexdigest()
                    result = self.enroll(person_id, decode_image(data), "upload", source_key)
                    job["accepted"] += int(result is not None)
                except (ValueError, RuntimeError) as exc:
                    job["errors"].append(f"{name}/{filename}: {exc}")
                job["done"] += 1
        finally:
            if archive:
                archive.close()

    def analyze(self, frame, session_id, mode="monitor"):
        self.require_ready()
        if not self.inference_lock.acquire(blocking=False):
            return None
        try:
            started = time.monotonic()
            self.refresh_gallery()
            for key in list(self.sessions):
                if started - self.sessions[key]["last_seen"] > 120:
                    if self.sessions[key].get("pose"):
                        self.sessions[key]["pose"].close()
                    del self.sessions[key]
            if session_id not in self.sessions:
                if len(self.sessions) >= 8:
                    raise ValueError("Too many active camera sessions")
                self.sessions[session_id] = {"tracks": [], "last_seen": started, "pose": None,
                    "fsm": FallStateMachine(studio_fall_config(self.settings["confirm_seconds"])),
                    "alerted": False, "next_track": 1}
            session = self.sessions[session_id]
            if started - session["last_seen"] > 1.5:
                session["fsm"].reset()
                session["tracks"] = []
                session["alerted"] = False
                if session["pose"]:
                    session["pose"].reset()
            session["last_seen"] = started
            found = self.engine.detect(frame)
            tracks, used = [], set()
            for face in found:
                pairs = [(FaceTracker._iou(face["bbox"], t["bbox"]), i)
                         for i, t in enumerate(session["tracks"]) if i not in used]
                overlap, index = max(pairs, default=(0, -1))
                if overlap >= .25:
                    track = session["tracks"][index]
                    used.add(index)
                else:
                    track = {"track_id": session["next_track"], "last_embed": -100,
                             "candidate": None, "hits": 0, "name": "Checking", "person_id": None}
                    session["next_track"] += 1
                track.update(bbox=face["bbox"], quality=face["quality"], warnings=face["warnings"])
                if mode == "enroll":
                    track.update(name="Ready to capture" if not face["warnings"] else "Adjust photo", person_id=None,
                                 similarity=None, reason=". ".join(face["warnings"]) or "Clear single-face sample")
                elif face["warnings"]:
                    track.update(name="Checking", person_id=None, similarity=None, reason=". ".join(face["warnings"]), hits=0)
                elif started - track["last_embed"] >= .65:
                    result = match_face(self.engine.embed(face["aligned"]) if self.gallery else None, self.gallery,
                                        threshold=self.settings["similarity_threshold"])
                    track["last_embed"] = started
                    candidate = result["person_id"]
                    track["hits"] = track["hits"] + 1 if candidate and candidate == track["candidate"] else 1
                    track["candidate"] = candidate
                    track.update(result)
                    if candidate and track["hits"] < 2:
                        track.update(name="Checking", person_id=None, reason="Confirming identity")
                tracks.append(track)
            session["tracks"] = tracks
            fall = {"state": "NORMAL", "pose_usable": False, "score": 0, "signals": {}, "landmarks": []}
            if mode == "monitor":
                if session["pose"] is None:
                    if self.mp is None:
                        import mediapipe as mp
                        self.mp = mp
                    session["pose"] = self.mp.solutions.pose.Pose(model_complexity=1, smooth_landmarks=True,
                        min_detection_confidence=.55, min_tracking_confidence=.55)
                result = session["pose"].process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                landmarks = result.pose_landmarks.landmark if result.pose_landmarks else None
                fsm = session["fsm"]
                state = fsm.update_rich(landmarks, *frame.shape[:2], started)
                fall = {"state": state, "pose_usable": fsm.last_pose_usable,
                        "score": round(fsm.last_score, 2), "signals": fsm.last_breakdown,
                        "landmarks": [[round(p.x, 4), round(p.y, 4), round(p.visibility, 2)] for p in landmarks] if landmarks else []}
                if state == CONFIRMED_FALL and not session["alerted"]:
                    who = "Unassigned person"
                    if landmarks and landmarks[0].visibility >= .5:
                        nose_x, nose_y = landmarks[0].x * frame.shape[1], landmarks[0].y * frame.shape[0]
                        associated = [t for t in tracks if t["person_id"] and t["bbox"][0] <= nose_x <= t["bbox"][0]+t["bbox"][2]
                                      and t["bbox"][1] <= nose_y <= t["bbox"][1]+t["bbox"][3]]
                        if len(associated) == 1:
                            who = associated[0]["name"]
                    self.registry.event("FALL", who, {"score": fall["score"], "signals": fall["signals"]})
                    session["alerted"] = True
                elif state != CONFIRMED_FALL:
                    session["alerted"] = False
            public_tracks = [{k: v for k, v in t.items() if k not in ("candidate", "hits", "last_embed")} for t in tracks]
            return {"faces": public_tracks, "fall": fall, "width": frame.shape[1], "height": frame.shape[0],
                    "processing_ms": round((time.monotonic() - started) * 1000), "trained_people": len(self.gallery)}
        finally:
            self.inference_lock.release()
