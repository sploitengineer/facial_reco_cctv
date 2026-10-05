"""Extract cached pose features, tune thresholds, and evaluate labeled video clips."""

import argparse
from contextlib import closing
import csv
from dataclasses import asdict, replace
import itertools
import json
import math
from pathlib import Path

from .fall_detection import DEFAULT_CONFIG, FallConfig, FallStateMachine, CONFIRMED_FALL, posture_features
from .recordings import recording_frames


def extract(manifest_path, output):
    import cv2
    import mediapipe as mp
    if not hasattr(mp, "solutions"):
        raise RuntimeError("This prototype requires mediapipe==0.10.21 on Python 3.11/3.12")
    manifest = Path(manifest_path).resolve()
    with manifest.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Manifest must contain labeled clips")
    clips, seen = [], set()
    for row in rows:
        if row.get("label") not in ("fall", "adl") or row.get("split") not in ("train", "validation", "test"):
            raise ValueError("Each manifest row needs path,label (fall/adl),split (train/validation/test)")
        video = (manifest.parent / row["path"]).resolve()
        if video in seen:
            raise ValueError(f"Duplicate video across splits: {video}")
        seen.add(video)
        timings = (manifest.parent / row["timestamps"]).resolve() if row.get("timestamps") else None
        frames = []
        # Fresh tracker per clip prevents tracking state leaking between videos.
        with closing(recording_frames(video, timings)) as recording, mp.solutions.pose.Pose(
                min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            for frame, timestamp in recording:
                result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                landmarks = result.pose_landmarks.landmark if result.pose_landmarks else None
                features = posture_features(landmarks, *frame.shape[:2], DEFAULT_CONFIG.min_visibility)
                frames.append({"timestamp": timestamp,
                               "aspect_ratio": features[0] if features else None,
                               "torso_y": features[1] if features else None})
        if not frames:
            raise ValueError(f"Recording has no readable frames: {video}")
        clips.append({"path": str(video), "label": row["label"], "split": row["split"], "frames": frames})
        print(f"Extracted {len(frames)} frames: {video.name}", flush=True)
    Path(output).write_text(json.dumps({"schema_version": 1, "min_visibility": DEFAULT_CONFIG.min_visibility,
                                        "clips": clips}, indent=2, allow_nan=False), encoding="utf-8")


def load_features(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or data.get("min_visibility") != DEFAULT_CONFIG.min_visibility:
        raise ValueError("Unsupported feature schema or visibility threshold; re-extract features")
    clips = data.get("clips", [])
    if not clips:
        raise ValueError("Feature file contains no clips")
    seen = set()
    for clip in clips:
        if clip["path"] in seen:
            raise ValueError("Duplicate clip paths are not allowed")
        seen.add(clip["path"])
        if clip["label"] not in ("fall", "adl") or clip["split"] not in ("train", "validation", "test"):
            raise ValueError("Invalid clip label or split")
        if not clip["frames"]:
            raise ValueError("Every clip must contain frames")
        previous = -1.0
        for frame in clip["frames"]:
            timestamp = frame["timestamp"]
            if not math.isfinite(timestamp) or timestamp < 0 or timestamp <= previous:
                raise ValueError("Frame timestamps must be finite, nonnegative and strictly increasing")
            previous = timestamp
            aspect, torso_y = frame["aspect_ratio"], frame["torso_y"]
            if (aspect is None) != (torso_y is None):
                raise ValueError("Missing pose must have both features set to null")
            if aspect is not None and (not math.isfinite(aspect) or aspect < 0 or not math.isfinite(torso_y)):
                raise ValueError("Pose features must be finite and aspect ratio nonnegative")
    return clips


def evaluate(clips, config):
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    predictions = []
    for clip in clips:
        fsm = FallStateMachine(config)
        first_alert = None
        for frame in clip["frames"]:
            down = (frame["aspect_ratio"] is not None
                    and frame["aspect_ratio"] > config.aspect_ratio_threshold
                    and frame["torso_y"] > config.low_position_threshold)
            if fsm.update(down, frame["timestamp"]) == CONFIRMED_FALL and first_alert is None:
                first_alert = frame["timestamp"]
        predicted = first_alert is not None
        actual = clip["label"] == "fall"
        counts[("tp" if actual else "fp") if predicted else ("fn" if actual else "tn")] += 1
        predictions.append({"path": clip["path"], "label": clip["label"],
                            "predicted_fall": predicted, "first_alert_seconds": first_alert})
    tp, fp, tn, fn = (counts[k] for k in ("tp", "fp", "tn", "fn"))
    return {**counts, "clip_count": len(clips), "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "specificity": tn / (tn + fp) if tn + fp else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "predictions": predictions}


def tune(clips, aspects, positions, durations):
    training = [c for c in clips if c["split"] == "train"]
    if {c["label"] for c in training} != {"fall", "adl"}:
        raise ValueError("Training split requires both fall and ADL clips")
    candidates = []
    for aspect, position, duration in itertools.product(aspects, positions, durations):
        config = replace(DEFAULT_CONFIG, aspect_ratio_threshold=aspect,
                         low_position_threshold=position, confirm_seconds=duration)
        metrics = evaluate(training, config)
        candidates.append((config, metrics))
    if not candidates:
        raise ValueError("Threshold grid cannot be empty")
    # Prefer fewer false positives when F1 ties. Never select using held-out data.
    config, metrics = max(candidates, key=lambda item: (item[1]["f1"], -item[1]["fp"], item[1]["recall"]))
    report = {"selected_config": asdict(config), "selection_split": "train",
              "training_detected_falls": metrics["tp"] > 0,
              "candidates_evaluated": len(candidates), "train": metrics,
              "grid": [{"config": asdict(c), **{k: v for k, v in m.items() if k != "predictions"}}
                       for c, m in candidates]}
    for split in ("validation", "test"):
        subset = [c for c in clips if c["split"] == split]
        report[split] = evaluate(subset, config) if subset else None
    return config, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    extraction = sub.add_parser("extract", help="Run MediaPipe once and cache features")
    extraction.add_argument("manifest")
    extraction.add_argument("--output", default="data/evaluation/features.json")
    tuning = sub.add_parser("tune", help="Grid search cached features without ML dependencies")
    tuning.add_argument("features")
    tuning.add_argument("--aspects", type=float, nargs="+", default=[1.1, 1.3, 1.5])
    tuning.add_argument("--positions", type=float, nargs="+", default=[0.45, 0.55, 0.65])
    tuning.add_argument("--durations", type=float, nargs="+", default=[1.0, 2.0, 3.0])
    tuning.add_argument("--config-output", default="data/evaluation/fall_config.json")
    tuning.add_argument("--report", default="data/evaluation/report.json")
    args = parser.parse_args()
    try:
        if args.command == "extract":
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            extract(args.manifest, args.output)
        else:
            config, report = tune(load_features(args.features), args.aspects, args.positions, args.durations)
            Path(args.config_output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.report).parent.mkdir(parents=True, exist_ok=True)
            config.save(args.config_output)
            Path(args.report).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            print(f"Evaluated {report['candidates_evaluated']} settings; training F1={report['train']['f1']:.3f}")
            print(f"Config: {args.config_output}; report: {args.report}")
            if not report["training_detected_falls"]:
                print("WARNING: No training falls were detected anywhere in this grid. "
                      "The exported candidate is not suitable for deployment.")
            if report["validation"] is None and report["test"] is None:
                print("No held-out split supplied; these are training results only.")
    except (ValueError, KeyError, TypeError, OSError, ImportError, RuntimeError) as exc:
        parser.exit(1, f"Evaluation failed: {exc}\n")


if __name__ == "__main__":
    main()
