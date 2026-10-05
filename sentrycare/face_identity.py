"""Shared single-person recognition for the webcam and RTSP prototypes."""

import logging
from pathlib import Path
from . import paths  # Configure the local DeepFace model cache before its lazy import.

logger = logging.getLogger("SentryCare")


def identify_face(frame, db_path="known_faces", model_name="SFace", detector_backend="opencv", backend=None):
    """Return (name, integer bbox); None means no usable face or recognition failure.

    Detect a real face before matching: enforce_detection=False on a full frame
    can embed background as a face. An empty enrollment still permits unknown capture.
    This prototype selects one face and does not associate identities to multiple poses.
    """
    if backend is None:
        from deepface import DeepFace
        backend = DeepFace
    try:
        try:
            faces = backend.extract_faces(img_path=frame, detector_backend=detector_backend,
                                          enforce_detection=True)
        except ValueError as exc:
            logger.debug("No usable face: %s", exc)
            return None, None
        if not faces:
            return None, None
        area = max(faces, key=lambda f: f["facial_area"]["w"] * f["facial_area"]["h"])["facial_area"]
        height, width = frame.shape[:2]
        x, y = max(0, int(area["x"])), max(0, int(area["y"]))
        right, bottom = min(width, int(area["x"] + area["w"])), min(height, int(area["y"] + area["h"]))
        if right <= x or bottom <= y:
            return None, None
        bbox = (x, y, right - x, bottom - y)
        database = Path(db_path)
        enrolled = database.is_dir() and any(p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")
                                              for p in database.rglob("*") if p.is_file())
        if not enrolled:
            return "unknown", bbox
        results = backend.find(img_path=frame[y:bottom, x:right], db_path=str(database),
                               model_name=model_name, detector_backend=detector_backend,
                               enforce_detection=True, silent=True)
        if results and len(results[0]):
            return Path(results[0].iloc[0]["identity"]).parent.name, bbox
        return "unknown", bbox
    except Exception:
        logger.exception("Face recognition failed")
        return None, None
