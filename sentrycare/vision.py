"""One-pass YuNet detection, landmark alignment, and cached SFace matching."""

import hashlib
from pathlib import Path
import threading
import urllib.request

import cv2
import numpy as np

from .paths import DATA_DIR

MODEL_VERSION = "yunet-2023mar-sface-2021dec-aligned-v1"
YUNET_URL = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"


def normalized(vector):
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(vector)
    if vector.shape != (128,) or not np.isfinite(vector).all() or norm < 1e-9:
        raise ValueError("Invalid face embedding")
    return vector / norm


def match_face(vector, gallery, threshold=.45, margin=.06):
    """Raw cosine similarity, never presented as an accuracy probability."""
    scores = []
    for person in gallery:
        references = person["vectors"]
        if not len(references):
            continue
        similarities = sorted((float(np.dot(vector, ref)) for ref in references), reverse=True)
        score = float(np.mean(similarities[:min(2, len(similarities))]))
        scores.append((score, person["id"], person["name"]))
    scores.sort(reverse=True)
    if not scores:
        return {"name": "Unknown", "person_id": None, "similarity": None, "reason": "No trained profiles"}
    score, person_id, name = scores[0]
    ambiguous = len(scores) > 1 and score - scores[1][0] < margin
    known = score >= threshold and not ambiguous
    return {"name": name if known else "Unknown", "person_id": person_id if known else None,
            "similarity": round(score, 3), "reason": "Matched" if known else "Ambiguous match" if ambiguous else "No close match"}


class FaceEngine:
    def __init__(self, model_dir=None):
        self.model_dir = Path(model_dir or DATA_DIR / "models")
        self.lock = threading.RLock()
        self.detector = self.recognizer = None

    def initialize(self):
        self.model_dir.mkdir(parents=True, exist_ok=True)
        yunet = self.model_dir / "face_detection_yunet_2023mar.onnx"
        sface = self.model_dir / ".deepface/weights/face_recognition_sface_2021dec.onnx"
        for destination, url in ((yunet, YUNET_URL), (sface, SFACE_URL)):
            if destination.is_file() and destination.stat().st_size > 10000:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            partial = destination.with_suffix(".download")
            with urllib.request.urlopen(url, timeout=60) as source, partial.open("wb") as output:
                while block := source.read(1024 * 1024):
                    output.write(block)
            partial.replace(destination)
        if hashlib.sha256(yunet.read_bytes()).hexdigest() != "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4":
            raise RuntimeError("YuNet model checksum does not match the expected release")
        with self.lock:
            cv2.setNumThreads(2)
            self.detector = cv2.FaceDetectorYN.create(str(yunet), "", (640, 480), .85, .3, 100)
            self.recognizer = cv2.FaceRecognizerSF.create(str(sface), "")

    def detect(self, frame):
        with self.lock:
            if self.detector is None:
                raise RuntimeError("Face engine is still starting")
            height, width = frame.shape[:2]
            scale = min(1., 640 / width)
            small = cv2.resize(frame, (round(width * scale), round(height * scale))) if scale < 1 else frame
            self.detector.setInputSize((small.shape[1], small.shape[0]))
            _, detections = self.detector.detect(small)
            result = []
            for detected in ([] if detections is None else detections):
                row = detected.copy()
                row[:14] /= scale
                aligned = self.recognizer.alignCrop(frame, row)
                x, y, w, h = row[:4]
                x, y = max(0, int(x)), max(0, int(y))
                w, h = min(width - x, int(w)), min(height - y, int(h))
                if w <= 0 or h <= 0:
                    continue
                gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
                quality = {"sharpness": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1),
                           "brightness": round(float(gray.mean()), 1), "face_pixels": min(w, h),
                           "detector_score": round(float(row[-1]), 3)}
                warnings = []
                if min(w, h) < 70:
                    warnings.append("Move closer: face is too small")
                if quality["sharpness"] < 30:
                    warnings.append("Hold still: photo is blurry")
                if quality["brightness"] < 40:
                    warnings.append("Face is too dark; face a light source")
                if quality["brightness"] > 225:
                    warnings.append("Face is overexposed; reduce the light")
                result.append({"bbox": [x, y, w, h], "aligned": aligned,
                               "quality": quality, "warnings": warnings})
            return result

    def embed(self, aligned):
        with self.lock:
            return normalized(self.recognizer.feature(aligned))
