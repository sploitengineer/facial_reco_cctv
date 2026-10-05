"""Dependency-free posture features and temporal fall confirmation."""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time

NORMAL, POSSIBLE_FALL, CONFIRMED_FALL = "NORMAL", "POSSIBLE_FALL", "CONFIRMED_FALL"


@dataclass(frozen=True)
class FallConfig:
    aspect_ratio_threshold: float = 1.3
    low_position_threshold: float = 0.55
    confirm_seconds: float = 3.0
    recover_frames: int = 8
    min_visibility: float = 0.5

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.aspect_ratio_threshold,
                   self.low_position_threshold, self.confirm_seconds, self.min_visibility)):
            raise ValueError("Thresholds must be finite")
        if self.aspect_ratio_threshold <= 0 or self.confirm_seconds <= 0:
            raise ValueError("Aspect ratio and confirmation duration must be positive")
        if not 0 <= self.low_position_threshold <= 1 or not 0 <= self.min_visibility <= 1:
            raise ValueError("Position and visibility thresholds must be between 0 and 1")
        if isinstance(self.recover_frames, bool) or not isinstance(self.recover_frames, int) or self.recover_frames < 1:
            raise ValueError("recover_frames must be a positive integer")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")


DEFAULT_CONFIG = FallConfig()


def posture_features(landmarks, frame_h, frame_w, min_visibility=0.5):
    """Return (torso aspect ratio, normalized height), or None for unreliable pose."""
    if frame_h <= 0 or frame_w <= 0:
        raise ValueError("Frame dimensions must be positive")
    if landmarks is None or len(landmarks) < 25:
        return None
    pts = [landmarks[i] for i in (11, 12, 23, 24)]
    if any(not all(math.isfinite(v) for v in (p.x, p.y, p.visibility))
           or p.visibility < min_visibility for p in pts):
        return None
    xs, ys = [p.x for p in pts], [p.y for p in pts]
    height = (max(ys) - min(ys)) * frame_h
    if height <= 1e-6:
        return None
    return (max(xs) - min(xs)) * frame_w / height, sum(ys) / 4


def is_down_posture(landmarks, frame_h, frame_w, config=DEFAULT_CONFIG):
    features = posture_features(landmarks, frame_h, frame_w, config.min_visibility)
    return bool(features and features[0] > config.aspect_ratio_threshold
                and features[1] > config.low_position_threshold)


class FallStateMachine:
    def __init__(self, config=DEFAULT_CONFIG):
        self.config = config
        self.reset()

    def reset(self):
        self.state = NORMAL
        self.down_since = None
        self.recover_streak = 0
        self.last_timestamp = None

    def update(self, down_now: bool, timestamp=None) -> str:
        """Use monotonic live time or explicit video seconds for offline evaluation.

        An interruption restarts confirmation; recovery still uses consecutive frames.
        Missing/unreliable poses should be passed as False.
        """
        now = time.monotonic() if timestamp is None else timestamp
        if not math.isfinite(now) or (self.last_timestamp is not None and now < self.last_timestamp):
            raise ValueError("Timestamps must be finite and nondecreasing")
        self.last_timestamp = now
        if down_now:
            self.recover_streak = 0
            if self.state == NORMAL:
                self.state = POSSIBLE_FALL
            if self.down_since is None:
                self.down_since = now
            if self.state == POSSIBLE_FALL and now - self.down_since >= self.config.confirm_seconds:
                self.state = CONFIRMED_FALL
        else:
            self.down_since = None
            if self.state != NORMAL:
                self.recover_streak += 1
                if self.recover_streak >= self.config.recover_frames:
                    self.state = NORMAL
                    self.recover_streak = 0
        return self.state
