"""Face recognition helpers: single-person and multi-person variants with improved accuracy."""

import logging
from pathlib import Path
import numpy as np
from . import paths  # Configure the local DeepFace model cache before its lazy import.

logger = logging.getLogger("SentryCare")

# Distance thresholds per model for rejecting weak matches.
# Tuned values: slightly relaxed from DeepFace defaults to reduce false "unknown"
# labels while still rejecting genuinely wrong matches.
_DISTANCE_THRESHOLDS = {
    "SFace": 0.593,      # DeepFace's tuned SFace cosine threshold
    "VGG-Face": 0.45,
    "Facenet": 0.45,
    "Facenet512": 0.35,
    "ArcFace": 0.68,
    "Dlib": 0.10,
    "OpenFace": 0.15,
}

# Minimum face confidence from detector to keep a detection.
_MIN_FACE_CONFIDENCE = 0.50
_MIN_IDENTITY_MARGIN = 0.04
_GALLERY_CACHE = {}


def _extract_with_detector(backend, frame, detector):
    try:
        return backend.extract_faces(
            img_path=frame, detector_backend=detector, enforce_detection=True
        ) or []
    except Exception as exc:
        logger.debug("Face detector %s failed: %s", detector, exc)
        return []


def _extract_faces(backend, frame, detector_backend):
    """Combine multiple local detectors, de-duplicating boxes for better recall.

    Uses the primary detector, a fallback, and optionally retinaface for
    superior multi-face detection (catches faces missed by single detectors).
    """
    fallback = "opencv" if detector_backend != "opencv" else "mediapipe"
    candidates = _extract_with_detector(backend, frame, detector_backend)
    if not candidates:
        candidates = _extract_with_detector(backend, frame, fallback)
    # Fallbacks run only on a miss; Studio uses one native YuNet pass.
    merged = []
    for face in candidates:
        area = face.get("facial_area", {})
        box = (area.get("x", 0), area.get("y", 0), area.get("w", 0), area.get("h", 0))
        if box[2] <= 0 or box[3] <= 0:
            continue
        duplicate = None
        for index, existing in enumerate(merged):
            other = existing["facial_area"]
            other_box = (other["x"], other["y"], other["w"], other["h"])
            ix1, iy1 = max(box[0], other_box[0]), max(box[1], other_box[1])
            ix2 = min(box[0] + box[2], other_box[0] + other_box[2])
            iy2 = min(box[1] + box[3], other_box[1] + other_box[3])
            intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            smaller_area = min(box[2] * box[3], other_box[2] * other_box[3])
            if smaller_area and intersection / smaller_area >= 0.65:
                duplicate = index
                break
        if duplicate is None:
            merged.append(face)
        else:
            existing_conf = float(merged[duplicate].get("confidence", 0) or 0)
            new_conf = float(face.get("confidence", 0) or 0)
            if new_conf > existing_conf:
                merged[duplicate] = face
    return sorted(merged, key=lambda f: f["facial_area"]["x"])


def _embedding_from_face(face, backend, model_name):
    """Embed an already detected/aligned RGB face crop (not its full frame)."""
    crop = np.asarray(face["face"])
    if crop.ndim != 3 or crop.shape[2] != 3 or crop.size == 0:
        raise ValueError("Detector returned an invalid face crop")
    if np.issubdtype(crop.dtype, np.floating) and crop.max(initial=0) <= 1.0:
        crop = crop * 255.0
    # DeepFace accepts BGR arrays; extract_faces returns normalized RGB arrays.
    crop = np.clip(crop, 0, 255).astype(np.uint8)[:, :, ::-1].copy()
    rows = backend.represent(
        img_path=crop,
        model_name=model_name,
        detector_backend="skip",
        enforce_detection=False,
        align=False,
    )
    vector = np.asarray(rows[0]["embedding"], dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Face model returned an invalid embedding")
    return vector / norm


def _gallery_signature(database):
    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return tuple(
        (str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(database.rglob("*"))
        if path.is_file() and path.suffix.lower() in image_suffixes
    )


def _load_gallery(db_path, model_name, detector_backend, backend):
    database = Path(db_path)
    if not database.is_dir():
        return {}
    # Gallery embeddings come from images; old pickle caches are not loaded or removed.
    signature = _gallery_signature(database)
    cache_key = (str(database.resolve()), model_name, detector_backend, id(backend))
    cached = _GALLERY_CACHE.get(cache_key)
    if cached and cached[0] == signature:
        return cached[1]

    gallery = {}
    for image_path, _, _ in signature:
        try:
            # Require an unambiguous single-person enrollment image. Group
            # photos cannot safely be assigned to the folder's identity.
            detected = _extract_faces(backend, image_path, detector_backend)
            if len(detected) > 1:
                logger.warning("Skipping multi-face enrollment photo: %s", Path(image_path).name)
                continue
            face = detected[0] if len(detected) == 1 else None
            if face is None or face.get("confidence", 1.0) < _MIN_FACE_CONFIDENCE:
                logger.warning("Skipping enrollment photo without a clear face: %s", Path(image_path).name)
                continue
            person = Path(image_path).parent.name
            gallery.setdefault(person, []).append(_embedding_from_face(face, backend, model_name))
        except Exception:
            logger.warning("Could not use enrollment photo %s", Path(image_path).name, exc_info=True)

    _GALLERY_CACHE[cache_key] = (signature, gallery)
    if not gallery:
        logger.warning("No usable single-face enrollment photos found in %s", database)
    else:
        logger.info("Loaded face gallery for %d enrolled identities", len(gallery))
        expected = {Path(path).parent.name for path, _, _ in signature}
        missing = sorted(expected - gallery.keys())
        if missing:
            logger.warning("No clean single-face enrollment photo for: %s", ", ".join(missing))
        for person, vectors in sorted(gallery.items()):
            if len(vectors) < 3:
                logger.warning(
                    "%s has only %d usable enrollment photo(s); add 3-5 clear single-person photos",
                    person, len(vectors),
                )
    return gallery


def _match_embedding(embedding, gallery, model_name):
    if not gallery:
        return "unknown"
    distances = sorted(
        (min(1.0 - float(np.dot(embedding, enrolled)) for enrolled in references), name)
        for name, references in gallery.items()
    )
    best_distance, best_name = distances[0]
    threshold = _DISTANCE_THRESHOLDS.get(model_name, 0.50)
    if best_distance > threshold:
        return "unknown"
    if len(distances) > 1 and distances[1][0] - best_distance < _MIN_IDENTITY_MARGIN:
        return "unknown"
    return best_name


def identify_face(frame, db_path="known_faces", model_name="SFace", detector_backend="mediapipe", backend=None):
    """Return (name, integer bbox); None means no usable face or recognition failure.

    Detect a real face before matching: enforce_detection=False on a full frame
    can embed background as a face. An empty enrollment still permits unknown capture.
    This compatibility helper selects the largest detected face. Use
    :func:`identify_all_faces` when a frame may contain multiple people.
    """
    faces = identify_all_faces(
        frame, db_path=db_path, model_name=model_name,
        detector_backend=detector_backend, backend=backend,
    )
    return max(faces, key=lambda item: item[1][2] * item[1][3]) if faces else (None, None)


def identify_all_faces(frame, db_path="known_faces", model_name="SFace", detector_backend="mediapipe", backend=None):
    """Return a list of (name, bbox) for every detected face in *frame*.

    Each bbox is an (x, y, w, h) integer tuple.  If the enrollment database
    is empty every face is labelled ``"unknown"``.  Recognition failures for
    individual crops are logged and skipped so that one bad crop does not
    prevent other faces from being reported.

    Uses confidence filtering and distance-gated matching for accuracy.
    """
    if backend is None:
        from deepface import DeepFace
        backend = DeepFace
    try:
        faces = _extract_faces(backend, frame, detector_backend)
        if not faces:
            return []
        confident_faces = [f for f in faces if f.get("confidence", 1.0) >= _MIN_FACE_CONFIDENCE]
        if not confident_faces:
            # No face passed the confidence filter — only fall back if the
            # detector returned exactly one face (ambiguity is low) and its
            # confidence is at least half the threshold. This avoids flooding
            # the output with low-quality false detections in multi-face mode.
            if len(faces) == 1 and float(faces[0].get("confidence", 0) or 0) >= _MIN_FACE_CONFIDENCE * 0.5:
                confident_faces = faces
            else:
                return []

        height, width = frame.shape[:2]
        database = Path(db_path)
        gallery = _load_gallery(database, model_name, detector_backend, backend)
        identities = []
        for face in confident_faces:
            area = face["facial_area"]
            x, y = max(0, int(area["x"])), max(0, int(area["y"]))
            right = min(width, int(area["x"] + area["w"]))
            bottom = min(height, int(area["y"] + area["h"]))
            if right <= x or bottom <= y:
                continue
            bbox = (x, y, right - x, bottom - y)
            name = "unknown"
            if gallery:
                try:
                    embedding = _embedding_from_face(face, backend, model_name)
                    name = _match_embedding(embedding, gallery, model_name)
                except Exception:
                    logger.warning("Could not embed a detected face", exc_info=True)
            identities.append((name, bbox))
        return identities
    except Exception:
        logger.exception("Multi-face detection or recognition failed")
        return []


class FaceTracker:
    """Simple IoU-based tracker to carry face identities between detection frames.

    Associates detections across frames without allowing stale names to
    override the current recognition result.
    """

    def __init__(self, stale_timeout=2.0, iou_threshold=0.3):
        self._tracks = []       # List of {"name": str, "bbox": tuple, "last_seen": float}
        self._stale_timeout = stale_timeout
        self._iou_threshold = iou_threshold

    @staticmethod
    def _iou(box_a, box_b):
        """Compute intersection over union for two (x, y, w, h) boxes."""
        ax, ay, aw, ah = box_a
        bx, by, bw, bh = box_b
        x1 = max(ax, bx)
        y1 = max(ay, by)
        x2 = min(ax + aw, bx + bw)
        y2 = min(ay + ah, by + bh)
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter == 0:
            return 0.0
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    def update(self, detections, timestamp):
        """Update tracks with new detections.

        Args:
            detections: list of (name, bbox) from identify_all_faces.
            timestamp: monotonic time.

        Returns:
            list of (name, bbox) with smoothed identities.
        """
        # Remove stale tracks.
        self._tracks = [t for t in self._tracks if timestamp - t["last_seen"] < self._stale_timeout]

        if not detections:
            # Keep tracks internally for a short re-detection gap, but don't
            # report absent faces as current detections.
            return []

        # Match globally by IoU so detector ordering cannot assign one prior
        # identity to multiple overlapping detections in the same frame.
        pairs = sorted(
            ((self._iou(bbox, track["bbox"]), di, ti)
             for di, (_, bbox) in enumerate(detections)
             for ti, track in enumerate(self._tracks)),
            reverse=True,
        )
        detection_tracks = {}
        used_tracks = set()
        for overlap, di, ti in pairs:
            if overlap < self._iou_threshold:
                break
            if di in detection_tracks or ti in used_tracks:
                continue
            detection_tracks[di] = ti
            used_tracks.add(ti)

        results = []
        updated_tracks = []
        for di, (name, bbox) in enumerate(detections):
            ti = detection_tracks.get(di)
            if ti is None:
                track = {"name": name or "unknown", "bbox": bbox, "last_seen": timestamp}
            else:
                track = self._tracks[ti]
                # Respect a fresh recognition result, including unknown.
                if name and name != "unknown":
                    track["name"] = name
                else:
                    # A fresh negative result must not inherit somebody else's name.
                    track["name"] = name or "unknown"
                track["bbox"] = bbox
                track["last_seen"] = timestamp
            updated_tracks.append(track)
            results.append((track["name"], bbox))

        # Keep unmatched tracks briefly for association, but never draw or alert
        # using a face that was absent from the current detection frame.
        self._tracks = updated_tracks + [
            track for ti, track in enumerate(self._tracks) if ti not in used_tracks
        ]
        return results

    def reset(self):
        self._tracks.clear()
