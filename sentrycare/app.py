"""SentryCare single-camera RTSP/webcam prototype with event logging and optional SMS."""

import argparse
from contextlib import closing
import csv
from datetime import datetime
import logging
import os
import json
from pathlib import Path
import time
from uuid import uuid4

from .face_identity import identify_face as recognize_face
from .fall_detection import FallConfig, FallStateMachine, is_down_posture, CONFIRMED_FALL
from .video_stream import frames
from .paths import PROJECT_ROOT, KNOWN_FACES, CAPTURED_FACES, EVENT_LOGS, LOGS, ensure_data_dirs

logger = logging.getLogger("SentryCare")
BASE_DIR = PROJECT_ROOT
DB_PATH = KNOWN_FACES
OUTPUT_DIR = CAPTURED_FACES
EVENT_LOG_DIR = EVENT_LOGS
MODEL_NAME = "SFace"
DETECTOR_BACKEND = "opencv"
FACE_ID_EVERY_N_FRAMES = 15


def identify_face(frame):
    return recognize_face(frame, DB_PATH, MODEL_NAME, DETECTOR_BACKEND)


def send_alert(who, event_type, frame=None, phone=None):
    import cv2
    now = datetime.now().astimezone()
    timestamp = now.isoformat(timespec="seconds")
    message = f"[ALERT] {event_type} - {who} at {timestamp}"
    logger.warning(message)
    EVENT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    if frame is not None:
        path = EVENT_LOG_DIR / f"{event_type}_{now:%Y%m%d_%H%M%S_%f}_{uuid4().hex[:8]}.jpg"
        if cv2.imwrite(str(path), frame):
            logger.info("Event snapshot saved: %s", path.name)
        else:
            logger.error("Could not save event snapshot")
    delivery = "console"
    if phone:
        try:
            from twilio.rest import Client
            credentials = [os.environ.get(k) for k in ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_FROM_NUMBER")]
            if all(credentials):
                Client(credentials[0], credentials[1]).messages.create(body=message, from_=credentials[2], to=phone)
                delivery = "twilio_accepted"
            else:
                delivery = "sms_unconfigured"
                logger.warning("Twilio credentials missing; SMS skipped")
        except ImportError:
            delivery = "sms_unavailable"
            logger.warning("Install twilio to enable SMS")
        except Exception:
            delivery = "sms_failed"
            logger.exception("SMS dispatch failed")
    log_file = EVENT_LOG_DIR / f"events_{now:%Y%m%d}.csv"
    needs_header = not log_file.exists() or log_file.stat().st_size == 0
    # Preserve the existing Phase 5 CSV schema, including logs created before this change.
    with log_file.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        if needs_header:
            writer.writerow(["timestamp", "event_type", "person", "alert_sent"])
        writer.writerow([timestamp, event_type, who, delivery])
    return delivery


def main(argv=None):
    global EVENT_LOG_DIR, OUTPUT_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="0", help="Webcam index, video file, or RTSP URL")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--alert-phone")
    parser.add_argument("--max-reconnect", type=int, default=5)
    parser.add_argument("--fall-config", help="JSON thresholds exported by phase6_evaluation.py")
    parser.add_argument("--no-face-id", action="store_true", help="Run pose detection without DeepFace")
    parser.add_argument("--camera-id", default="default", help="Safe camera name for separate event directories")
    parser.add_argument("--max-frames", type=int, help="Stop after this many frames (for smoke tests)")
    parser.add_argument("--heartbeat", help="Write atomic processing status for the supervisor")
    args = parser.parse_args(argv)
    if args.max_reconnect < 0:
        parser.error("--max-reconnect must be nonnegative")
    import re
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", args.camera_id):
        parser.error("--camera-id must contain 1-64 letters, digits, underscores or hyphens")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    try:
        config = FallConfig.load(args.fall_config) if args.fall_config else FallConfig()
        import cv2
        import mediapipe as mp
        if not hasattr(mp, "solutions"):
            parser.error("Use mediapipe==0.10.21 with Python 3.11/3.12; this prototype uses the legacy Pose API")
        if not args.no_face_id:
            from deepface import DeepFace  # Fail at startup instead of once per inference.
    except (ImportError, ValueError, TypeError, OSError) as exc:
        parser.error(str(exc))

    ensure_data_dirs()
    log_dir = LOGS
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(
                            log_dir / f"sentrycare_{args.camera_id}_{datetime.now():%Y%m%d}.log", encoding="utf-8")])
    EVENT_LOG_DIR = EVENT_LOGS / args.camera_id
    OUTPUT_DIR = CAPTURED_FACES / args.camera_id
    DB_PATH.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source = int(args.source) if args.source.isdigit() else args.source
    fsm = FallStateMachine(config)
    alerted, frame_count = False, 0
    last_name, last_bbox, last_identity_time = None, None, None
    last_capture = -float("inf")
    last_heartbeat = -float("inf")
    try:
        with closing(frames(source, args.max_reconnect)) as stream, mp.solutions.pose.Pose(
                min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            for frame, reconnected in stream:
                now = time.monotonic()
                frame_count += 1
                if reconnected:
                    fsm.reset()
                    alerted = False
                    last_name, last_bbox, last_identity_time = None, None, None
                    pose.reset()
                h, w = frame.shape[:2]
                results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                present = results.pose_landmarks is not None
                down = is_down_posture(results.pose_landmarks.landmark, h, w, config) if present else False
                state = fsm.update(down, now)
                if not present or (last_identity_time is not None and now - last_identity_time > 2.0):
                    last_name, last_bbox, last_identity_time = None, None, None
                if present and not args.no_face_id and frame_count % FACE_ID_EVERY_N_FRAMES == 0:
                    last_name, last_bbox = identify_face(frame)
                    last_identity_time = now if last_name else None
                    if last_name == "unknown" and now - last_capture >= 2.0 and last_bbox:
                        x, y, bw, bh = last_bbox
                        path = OUTPUT_DIR / f"unknown_{datetime.now():%Y%m%d_%H%M%S_%f}.jpg"
                        if cv2.imwrite(str(path), frame[y:y+bh, x:x+bw]):
                            last_capture = now
                if state == CONFIRMED_FALL and not alerted:
                    send_alert(last_name or "unidentified person", "FALL", frame, args.alert_phone)
                    alerted = True
                elif state != CONFIRMED_FALL:
                    alerted = False
                if args.heartbeat and now - last_heartbeat >= 1.0:
                    heartbeat = Path(args.heartbeat)
                    heartbeat.parent.mkdir(parents=True, exist_ok=True)
                    temp = heartbeat.with_suffix(".tmp")
                    temp.write_text(json.dumps({"camera_id": args.camera_id, "frame_count": frame_count,
                                               "state": state, "person_present": present}), encoding="utf-8")
                    temp.replace(heartbeat)
                    last_heartbeat = now
                if not args.headless:
                    if present:
                        mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
                    color = {"NORMAL": (0, 200, 0), "POSSIBLE_FALL": (0, 165, 255), "CONFIRMED_FALL": (0, 0, 255)}[state]
                    cv2.putText(frame, f"STATE: {state}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    if last_name:
                        cv2.putText(frame, f"ID: {last_name}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                        if last_bbox:
                            x, y, bw, bh = last_bbox
                            cv2.rectangle(frame, (x, y), (x+bw, y+bh), color, 2)
                    cv2.imshow("SentryCare - Phase 5", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                if args.max_frames is not None and frame_count >= args.max_frames:
                    break
    except KeyboardInterrupt:
        logger.info("Stopped by user")
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1
    finally:
        if not args.headless:
            cv2.destroyAllWindows()
        logger.info("SentryCare shutdown complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
