"""Improved posture features and temporal fall confirmation with multi-signal analysis."""

from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import time
from collections import deque

NORMAL, POSSIBLE_FALL, CONFIRMED_FALL = "NORMAL", "POSSIBLE_FALL", "CONFIRMED_FALL"


@dataclass(frozen=True)
class FallConfig:
    # --- Torso aspect-ratio signal (original, retuned) ---
    aspect_ratio_threshold: float = 1.3       # preserve baseline geometry defaults
    low_position_threshold: float = 0.55      # person in lower half of frame
    # --- Head-below-hips signal (NEW) ---
    head_hip_enable: bool = True
    head_hip_margin: float = 0.02             # head.y must exceed avg(hip.y) + margin (normalised)
    # --- Velocity signal (NEW) ---
    velocity_enable: bool = True
    velocity_threshold: float = 0.25          # normalised-y units/sec — a sudden downward move
    velocity_window: int = 5                  # rolling window (frames) for velocity estimation
    fall_motion_window: float = 2.0           # pair descent with the resulting low posture
    pose_gap_seconds: float = 0.25             # tolerate brief landmark dropouts during a fall
    # --- Temporal state machine ---
    confirm_seconds: float = 3.0              # baseline; Studio uses a shorter explicit window
    recover_frames: int = 8                  # more frames needed to declare recovery (avoid flicker)
    min_visibility: float = 0.50              # lenient visibility for partial poses
    # --- Scoring ---
    score_threshold: float = 1.3              # require a down posture plus fall evidence
    weight_aspect: float = 1.2                # aspect ratio is reliable
    weight_head_hip: float = 0.5               # corroborating signal, never sufficient alone
    weight_velocity: float = 0.8              # velocity is noisy; give less weight

    def __post_init__(self):
        finite_fields = (
            self.aspect_ratio_threshold, self.low_position_threshold,
            self.head_hip_margin, self.velocity_threshold, self.fall_motion_window,
            self.pose_gap_seconds, self.confirm_seconds,
            self.min_visibility, self.score_threshold,
            self.weight_aspect, self.weight_head_hip, self.weight_velocity,
        )
        if not all(math.isfinite(v) for v in finite_fields):
            raise ValueError("Thresholds must be finite")
        if (self.aspect_ratio_threshold <= 0 or self.confirm_seconds <= 0
                or self.fall_motion_window <= 0 or self.pose_gap_seconds < 0):
            raise ValueError("Aspect ratio and timing thresholds must be positive")
        if not 0 <= self.low_position_threshold <= 1 or not 0 <= self.min_visibility <= 1:
            raise ValueError("Position and visibility thresholds must be between 0 and 1")
        if isinstance(self.recover_frames, bool) or not isinstance(self.recover_frames, int) or self.recover_frames < 1:
            raise ValueError("recover_frames must be a positive integer")
        if isinstance(self.velocity_window, bool) or not isinstance(self.velocity_window, int) or self.velocity_window < 2:
            raise ValueError("velocity_window must be an integer >= 2")

    @classmethod
    def load(cls, path):
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        # Filter out any keys not in the dataclass to allow forward-compatible configs.
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in raw.items() if k in valid})

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")


DEFAULT_CONFIG = FallConfig()


def studio_fall_config(confirm_seconds=1.2):
    """UI cadence is about 5 Hz; use recent motion rather than a whole-second average."""
    return replace(FallConfig(), confirm_seconds=confirm_seconds, velocity_window=3)

# MediaPipe landmark indices used for full-body analysis.
_NOSE = 0
_LEFT_SHOULDER, _RIGHT_SHOULDER = 11, 12
_LEFT_HIP, _RIGHT_HIP = 23, 24
_LEFT_KNEE, _RIGHT_KNEE = 25, 26
_LEFT_ANKLE, _RIGHT_ANKLE = 27, 28


def _visible(point, min_vis):
    """Check whether a landmark has finite coords and enough visibility."""
    return (all(math.isfinite(v) for v in (point.x, point.y, point.visibility))
            and point.visibility >= min_vis)


def posture_features(landmarks, frame_h, frame_w, min_visibility=0.5):
    """Return (torso aspect ratio, normalized height), or None for unreliable pose.

    Uses original 4-point torso approach for backward compatibility.
    """
    if frame_h <= 0 or frame_w <= 0:
        raise ValueError("Frame dimensions must be positive")
    if landmarks is None or len(landmarks) < 25:
        return None
    pts = [landmarks[i] for i in (_LEFT_SHOULDER, _RIGHT_SHOULDER, _LEFT_HIP, _RIGHT_HIP)]
    if any(not _visible(p, min_visibility) for p in pts):
        return None
    xs, ys = [p.x for p in pts], [p.y for p in pts]
    height = (max(ys) - min(ys)) * frame_h
    if height <= 1e-6:
        return None
    return (max(xs) - min(xs)) * frame_w / height, sum(ys) / 4


def full_body_features(landmarks, frame_h, frame_w, config=DEFAULT_CONFIG):
    """Extract rich posture features for fall scoring.

    Returns a dict with keys:
      - torso_ar:     torso aspect ratio (width / height)
      - torso_y:      normalised average torso height (0 = top, 1 = bottom)
      - head_below_hips: True if the nose/head is below the average hip line
      - body_center_y:   normalised center of all visible body points
    Or None when pose data is unreliable.
    """
    if frame_h <= 0 or frame_w <= 0:
        raise ValueError("Frame dimensions must be positive")
    if landmarks is None or len(landmarks) < 29:
        return None
    min_vis = config.min_visibility

    # Core torso (must be visible).
    torso_ids = [_LEFT_SHOULDER, _RIGHT_SHOULDER, _LEFT_HIP, _RIGHT_HIP]
    torso_pts = [landmarks[i] for i in torso_ids]
    if any(not _visible(p, min_vis) for p in torso_pts):
        return None
    xs = [p.x for p in torso_pts]
    ys = [p.y for p in torso_pts]
    torso_h = (max(ys) - min(ys)) * frame_h
    if torso_h <= 1e-6:
        return None
    torso_ar = (max(xs) - min(xs)) * frame_w / torso_h
    torso_y = sum(ys) / len(ys)
    shoulder_center_x = (landmarks[_LEFT_SHOULDER].x + landmarks[_RIGHT_SHOULDER].x) / 2
    shoulder_center_y = (landmarks[_LEFT_SHOULDER].y + landmarks[_RIGHT_SHOULDER].y) / 2
    hip_center_x = (landmarks[_LEFT_HIP].x + landmarks[_RIGHT_HIP].x) / 2
    hip_center_y = (landmarks[_LEFT_HIP].y + landmarks[_RIGHT_HIP].y) / 2
    torso_axis_h = abs(shoulder_center_y - hip_center_y) * frame_h
    torso_axis_w = abs(shoulder_center_x - hip_center_x) * frame_w
    torso_axis_ar = torso_axis_w / max(torso_axis_h, 1.0)

    # Head position relative to hips.
    head_below_hips = False
    nose = landmarks[_NOSE]
    hip_avg_y = (landmarks[_LEFT_HIP].y + landmarks[_RIGHT_HIP].y) / 2
    if _visible(nose, min_vis * 0.7):  # more lenient for head
        head_below_hips = nose.y > (hip_avg_y + config.head_hip_margin)

    # Full body center (include legs if visible).
    all_ids = torso_ids + [_NOSE, _LEFT_KNEE, _RIGHT_KNEE, _LEFT_ANKLE, _RIGHT_ANKLE]
    visible_ys = [landmarks[i].y for i in all_ids if _visible(landmarks[i], min_vis * 0.5)]
    body_center_y = sum(visible_ys) / len(visible_ys) if visible_ys else torso_y

    return {
        "torso_ar": torso_ar,
        "torso_axis_ar": torso_axis_ar,
        "torso_y": torso_y,
        "head_below_hips": head_below_hips,
        "body_center_y": body_center_y,
    }


def is_down_posture(landmarks, frame_h, frame_w, config=DEFAULT_CONFIG):
    """Simple boolean check — backward compatible.

    Uses the original torso aspect-ratio + position logic.
    """
    features = posture_features(landmarks, frame_h, frame_w, config.min_visibility)
    return bool(features and features[0] > config.aspect_ratio_threshold
                and features[1] > config.low_position_threshold)


def compute_fall_score(features, velocity, config=DEFAULT_CONFIG, fall_motion=False):
    """Combine multiple signals into a single fall score.

    Returns (score, breakdown_dict).  score >= config.score_threshold → "down".
    """
    score = 0.0
    breakdown = {}

    # A low, wide pose needs support from a recent descent or an inverted
    # head/hip relationship. This prevents isolated torso/head signals firing.
    # However, a very extreme aspect ratio (person clearly horizontal) should
    # still contribute even without velocity/head evidence, just at reduced
    # weight, to avoid missing prolonged ground-level falls.
    head_below = config.head_hip_enable and features["head_below_hips"]
    down_pose = _is_down_pose(features, config)
    axis_ratio = features.get("torso_axis_ar", features["torso_ar"])
    posture_ratio = axis_ratio if axis_ratio > config.aspect_ratio_threshold else features["torso_ar"]
    corroborated = fall_motion or head_below
    extreme_ratio = posture_ratio > config.aspect_ratio_threshold * 1.5  # clearly horizontal
    if down_pose and (corroborated or extreme_ratio):
        s = min((posture_ratio / config.aspect_ratio_threshold - 1) + 0.5, 1.5)
        if not corroborated:
            s *= 0.6  # reduce weight when not corroborated by other signals
        score += s * config.weight_aspect
        breakdown["aspect_ratio"] = round(s, 2)

    # The reduced uncorroborated score is capped at 1.08 with default weights,
    # below the 1.3 gate: even sustained, clearly horizontal poses were unreachable.
    # Permit strong torso-axis evidence, still requiring the low-posture gate and
    # temporal confirmation. Ordinary shoulder width is not torso orientation.
    if down_pose and axis_ratio > config.aspect_ratio_threshold * 2:
        score = max(score, config.score_threshold)
        breakdown["horizontal_torso"] = 1.0

    # Signal 2: head below hips.
    if head_below and down_pose:
        score += 1.0 * config.weight_head_hip
        breakdown["head_below_hips"] = 1.0

    # Signal 3: sudden vertical velocity (downward).
    # Only contribute when we actually measured a velocity spike, not just
    # because the fall_motion flag was latched from an earlier frame.
    if config.velocity_enable and down_pose and velocity is not None and velocity > config.velocity_threshold:
        s = min(velocity / config.velocity_threshold, 2.0)
        score += s * config.weight_velocity
        breakdown["velocity"] = round(s, 2)
    elif config.velocity_enable and down_pose and fall_motion:
        # Latched fall motion (rapid descent was seen recently) but no
        # current velocity spike — add a reduced contribution so
        # confirmation can still complete after the person has stopped
        # moving but is lying on the ground.
        score += 0.5 * config.weight_velocity
        breakdown["velocity"] = 0.5

    return score, breakdown


def _is_down_pose(features, config):
    """Detect a low, sideways torso while retaining upside-down pose support."""
    low_posture = features["torso_y"] > config.low_position_threshold
    axis_ratio = features.get("torso_axis_ar", features["torso_ar"])
    sideways = axis_ratio > config.aspect_ratio_threshold
    inverted_wide_pose = (config.head_hip_enable and features["head_below_hips"]
                          and features["torso_ar"] > config.aspect_ratio_threshold)
    return low_posture and (sideways or inverted_wide_pose)


class VelocityTracker:
    """Track normalised-Y center of body across frames to estimate vertical velocity."""

    def __init__(self, window=5):
        self._history = deque(maxlen=max(window, 2))

    def update(self, center_y, timestamp):
        self._history.append((center_y, timestamp))
        if len(self._history) < 2:
            return None
        oldest_y, oldest_t = self._history[0]
        dt = timestamp - oldest_t
        if dt <= 1e-6:
            return None
        # Positive = moving downward (y increases in image coords).
        return (center_y - oldest_y) / dt

    def reset(self):
        self._history.clear()


class FallStateMachine:
    def __init__(self, config=DEFAULT_CONFIG):
        self.config = config
        self._velocity_tracker = VelocityTracker(config.velocity_window)
        self.reset()

    def reset(self):
        self.state = NORMAL
        self.down_since = None
        self.recover_streak = 0
        self.last_timestamp = None
        self.last_score = 0.0
        self.last_breakdown = {}
        self.last_pose_usable = False
        self._fall_motion_until = None
        self._fall_candidate_latched = False
        self._last_down_evidence = None
        self._velocity_tracker.reset()

    def update(self, down_now: bool, timestamp=None, pose_missing=False) -> str:
        """Use monotonic live time or explicit video seconds for offline evaluation.

        An interruption restarts confirmation; recovery still uses consecutive frames.
        Missing/unreliable poses should be passed as False.
        """
        now = time.monotonic() if timestamp is None else timestamp
        if not math.isfinite(now) or (self.last_timestamp is not None and now < self.last_timestamp):
            raise ValueError("Timestamps must be finite and nondecreasing")
        self.last_timestamp = now
        if pose_missing:
            if self.state == CONFIRMED_FALL:
                self.recover_streak = 0
                return self.state
            if (self.state == POSSIBLE_FALL and self.down_since is not None
                    and self._last_down_evidence is not None
                    and now - self._last_down_evidence <= self.config.pose_gap_seconds):
                self.recover_streak = 0
                return self.state
        if down_now:
            self.recover_streak = 0
            self._last_down_evidence = now
            if self.state == NORMAL:
                self.state = POSSIBLE_FALL
            if self.down_since is None:
                self.down_since = now
            if self.state == POSSIBLE_FALL and now - self.down_since >= self.config.confirm_seconds:
                self.state = CONFIRMED_FALL
        else:
            self.down_since = None
            self._last_down_evidence = None
            if self.state != NORMAL:
                self.recover_streak += 1
                if self.recover_streak >= self.config.recover_frames:
                    self.state = NORMAL
                    self.recover_streak = 0
        return self.state

    def update_rich(self, landmarks, frame_h, frame_w, timestamp=None) -> str:
        """Enhanced update using full-body multi-signal analysis.

        Feeds landmarks directly and internally computes fall score from
        aspect ratio + head position + velocity signals.
        """
        now = time.monotonic() if timestamp is None else timestamp
        features = full_body_features(landmarks, frame_h, frame_w, self.config)

        if features is None:
            # Unreliable pose — treat as not-down (don't advance timer).
            self.last_pose_usable = False
            self.last_score = 0.0
            self.last_breakdown = {}
            state = self.update(False, now, pose_missing=True)
            if state == NORMAL:
                self._fall_motion_until = None
                self._fall_candidate_latched = False
            return state

        self.last_pose_usable = True
        velocity = self._velocity_tracker.update(features["body_center_y"], now)
        if (self.config.velocity_enable and velocity is not None
                and velocity > self.config.velocity_threshold):
            self._fall_motion_until = now + self.config.fall_motion_window
        down_pose = _is_down_pose(features, self.config)
        recent_motion = self._fall_motion_until is not None and now <= self._fall_motion_until
        if recent_motion and down_pose:
            # Once descent is followed by a down posture, keep the evidence
            # latched until the person recovers so confirmation can complete.
            self._fall_candidate_latched = True
        fall_motion = recent_motion or self._fall_candidate_latched
        score, breakdown = compute_fall_score(features, velocity, self.config, fall_motion)
        self.last_score = score
        self.last_breakdown = breakdown
        down_now = score >= self.config.score_threshold
        state = self.update(down_now, now)
        if not down_pose and state == NORMAL:
            self._fall_candidate_latched = False
        return state
