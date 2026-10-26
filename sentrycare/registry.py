"""Local enrollment records, aligned sample images, embeddings, and event history."""

from datetime import datetime, timezone
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
import threading
from uuid import uuid4

import cv2
import numpy as np

from .vision import MODEL_VERSION, normalized


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Registry:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.images = self.root / "images"
        self.images.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "enrollment.sqlite3"
        self.lock = threading.RLock()
        self.version = 0
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS people (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, name_key TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS samples (
                    id TEXT PRIMARY KEY, person_id TEXT NOT NULL REFERENCES people(id) ON DELETE CASCADE,
                    angle TEXT NOT NULL, quality TEXT NOT NULL, created_at TEXT NOT NULL,
                    embedding BLOB, model TEXT, source_key TEXT UNIQUE);
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY, kind TEXT NOT NULL, person TEXT NOT NULL,
                    details TEXT NOT NULL, created_at TEXT NOT NULL);
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def people(self):
        with self.lock, self.connect() as db:
            result = []
            for person in db.execute("SELECT * FROM people ORDER BY name COLLATE NOCASE"):
                samples = []
                for row in db.execute("SELECT * FROM samples WHERE person_id=? ORDER BY created_at,id", (person["id"],)):
                    samples.append({"id": row["id"], "angle": row["angle"], "quality": json.loads(row["quality"]),
                                    "trained": row["embedding"] is not None and row["model"] == MODEL_VERSION,
                                    "image_url": f"/api/samples/{row['id']}/image"})
                trained = sum(s["trained"] for s in samples)
                result.append({"id": person["id"], "name": person["name"], "created_at": person["created_at"],
                               "samples": samples, "sample_count": len(samples), "trained_count": trained,
                               "status": "ready" if trained >= 3 and trained == len(samples) else
                                         "update_needed" if trained >= 3 else "needs_training" if len(samples) >= 3 else "needs_photos"})
            return result

    def person(self, person_id):
        return next((p for p in self.people() if p["id"] == person_id), None)

    def create_person(self, name):
        name = " ".join(str(name).split())
        if not 1 <= len(name) <= 60 or name.casefold() in ("unknown", "checking", "unidentified"):
            raise ValueError("Enter a name between 1 and 60 characters, other than Unknown")
        person_id = uuid4().hex
        with self.lock, self.connect() as db:
            try:
                db.execute("INSERT INTO people VALUES (?,?,?,?)", (person_id, name, name.casefold(), utc_now()))
            except sqlite3.IntegrityError as exc:
                raise ValueError("This name is already enrolled; select the existing profile") from exc
            self.version += 1
        return self.person(person_id)

    def save_sample(self, person_id, aligned, angle, quality, source_key=None):
        if angle not in ("front", "left", "right", "up", "down", "upload"):
            raise ValueError("Choose a valid photo angle")
        sample_id = uuid4().hex
        path = self.images / f"{sample_id}.jpg"
        with self.lock, self.connect() as db:
            if db.execute("SELECT id FROM people WHERE id=?", (person_id,)).fetchone() is None:
                raise ValueError("Profile no longer exists")
            count = db.execute("SELECT count(*) FROM samples WHERE person_id=?", (person_id,)).fetchone()[0]
            if count >= 30:
                raise ValueError("This profile already has 30 photos; remove redundant photos first")
            if source_key and db.execute("SELECT id FROM samples WHERE source_key=?", (source_key,)).fetchone():
                return None
            if not cv2.imwrite(str(path), aligned, [cv2.IMWRITE_JPEG_QUALITY, 96]):
                raise RuntimeError("Could not save enrollment image")
            db.execute("INSERT INTO samples VALUES (?,?,?,?,?,?,?,?)", (sample_id, person_id, angle,
                       json.dumps(quality), utc_now(), None, None, source_key))
            self.version += 1
        return sample_id

    def sample_rows(self, person_id=None):
        with self.connect() as db:
            query = "SELECT samples.*, people.name FROM samples JOIN people ON people.id=samples.person_id"
            rows = db.execute(query + (" WHERE person_id=?" if person_id else ""),
                              (person_id,) if person_id else ()).fetchall()
            return [dict(row) for row in rows]

    def save_embeddings(self, embeddings):
        with self.lock, self.connect() as db:
            for sample_id, vector in embeddings:
                db.execute("UPDATE samples SET embedding=?,model=? WHERE id=?",
                           (normalized(vector).tobytes(), MODEL_VERSION, sample_id))
            self.version += 1

    def gallery(self):
        with self.lock, self.connect() as db:
            people = {}
            for row in db.execute("SELECT samples.*,people.name FROM samples JOIN people ON people.id=samples.person_id "
                                  "WHERE embedding IS NOT NULL AND model=?", (MODEL_VERSION,)):
                person = people.setdefault(row["person_id"], {"id": row["person_id"], "name": row["name"], "vectors": []})
                try:
                    person["vectors"].append(normalized(np.frombuffer(row["embedding"], dtype=np.float32)))
                except ValueError:
                    continue
            return [p for p in people.values() if len(p["vectors"]) >= 3]

    def delete_sample(self, sample_id):
        with self.lock, self.connect() as db:
            changed = db.execute("DELETE FROM samples WHERE id=?", (sample_id,)).rowcount
            if changed:
                (self.images / f"{sample_id}.jpg").unlink(missing_ok=True)
                self.version += 1
            return bool(changed)

    def delete_person(self, person_id):
        with self.lock, self.connect() as db:
            ids = [r[0] for r in db.execute("SELECT id FROM samples WHERE person_id=?", (person_id,))]
            changed = db.execute("DELETE FROM people WHERE id=?", (person_id,)).rowcount
            for sample_id in ids:
                (self.images / f"{sample_id}.jpg").unlink(missing_ok=True)
            self.version += 1
            return bool(changed)

    def event(self, kind, person, details):
        with self.connect() as db:
            db.execute("INSERT INTO events VALUES (?,?,?,?,?)", (uuid4().hex, kind, person, json.dumps(details), utc_now()))

    def events(self):
        with self.connect() as db:
            return [{**dict(row), "details": json.loads(row["details"])}
                    for row in db.execute("SELECT * FROM events ORDER BY created_at DESC, rowid DESC LIMIT 100")]
