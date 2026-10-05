"""Read videos or URFD RGB image sequences using recorded frame times."""

import csv
import math
from pathlib import Path
import re


def sequence_entries(directory, timing_path):
    images = [p for p in Path(directory).rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
    if not images:
        raise ValueError(f"No RGB frames in {directory}")
    def number(path):
        groups = re.findall(r"\d+", path.stem)
        if not groups:
            raise ValueError(f"Image filename needs a frame number: {path.name}")
        return int(groups[-1])
    images.sort(key=number)
    times = {}
    # Official URFD cam0 sync CSV: frame number, milliseconds, accelerometer magnitude.
    with Path(timing_path).open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.reader(stream):
            if not row:
                continue
            index, timestamp = int(row[0]), float(row[1]) / 1000
            if index in times or not math.isfinite(timestamp) or timestamp < 0:
                raise ValueError("Invalid or duplicate sequence timing")
            times[index] = timestamp
    entries, seen, previous = [], set(), -1.0
    for image in images:
        index = number(image)
        if index in seen or index not in times or times[index] <= previous:
            raise ValueError(f"Missing, duplicate or unordered frame timing: {image.name}")
        seen.add(index)
        previous = times[index]
        entries.append((image, times[index]))
    return entries


def recording_frames(path, timestamps=None):
    import cv2
    path = Path(path)
    if path.is_dir():
        if timestamps is None:
            raise ValueError("Image sequences require a timestamps CSV (URFD cam0 format)")
        for image, timestamp in sequence_entries(path, timestamps):
            frame = cv2.imread(str(image))
            if frame is None:
                raise ValueError(f"Cannot decode image: {image}")
            yield frame, timestamp
        return
    if not path.is_file():
        raise ValueError(f"Recording does not exist: {path}")
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"Missing/invalid frame rate: {path}")
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame, index / fps
            index += 1
    finally:
        cap.release()
