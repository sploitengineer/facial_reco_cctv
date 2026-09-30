"""
Phase 5 — Production-ready prototype: RTSP stream + alerting.

Swaps the webcam (VideoCapture(0)) for an RTSP stream URL from an IP camera,
and replaces the print() fall alert with a real notification dispatch
(console + optional Twilio SMS / WhatsApp).

Usage:
  # With webcam (default, for testing):
  python phase5_production.py

  # With an RTSP camera:
  python phase5_production.py --source "rtsp://user:pass@192.168.1.100:554/stream1"

  # With SMS alerting enabled:
  python phase5_production.py --alert-phone "+919876543210"

Press 'q' to quit (only works if --headless is not set).
"""

import os
import sys
import time
import argparse
import logging
from datetime import datetime

import cv2
import mediapipe as mp
from deepface import DeepFace

from fall_detection_webcam import FallStateMachine, is_down_posture, CONFIRMED_FALL

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(LOG_DIR, f"sentrycare_{datetime.now():%Y%m%d}.log")),
    ],
)
logger = logging.getLogger("SentryCare")

# ---------------------------------------------------------------------------
# MediaPipe setup
# ---------------------------------------------------------------------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DB_PATH = "known_faces"
OUTPUT_DIR = "captured_faces"
EVENT_LOG_DIR = "event_logs"
MODEL_NAME = "SFace"
DETECTOR_BACKEND = "opencv"
FACE_ID_EVERY_N_FRAMES = 15

os.makedirs(DB_PATH, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EVENT_LOG_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Alerting
# ---------------------------------------------------------------------------
def send_alert(who: str, event_type: str, frame=None, phone: str = None):
    """
    Dispatch an alert for a confirmed event.
    Currently logs to console + event log file.
    Optionally sends SMS/WhatsApp via Twilio if credentials are configured.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = f"[ALERT] {event_type} — {who} at {timestamp}"
    logger.warning(message)

    # Save event log entry
    log_file = os.path.join(EVENT_LOG_DIR, f"events_{datetime.now():%Y%m%d}.csv")
    header_needed = not os.path.exists(log_file)
    with open(log_file, "a") as f:
        if header_needed:
            f.write("timestamp,event_type,person,alert_sent\n")
        f.write(f"{timestamp},{event_type},{who},console\n")

    # Save snapshot if frame is available
    if frame is not None:
        snapshot_path = os.path.join(EVENT_LOG_DIR, f"{event_type}_{datetime.now():%Y%m%d_%H%M%S}.jpg")
        cv2.imwrite(snapshot_path, frame)
        logger.info(f"Event snapshot saved: {snapshot_path}")

    # --- Twilio SMS/WhatsApp (optional) ---
    if phone:
        try:
            from twilio.rest import Client
            account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
            auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
            from_number = os.environ.get("TWILIO_FROM_NUMBER")

            if account_sid and auth_token and from_number:
                client = Client(account_sid, auth_token)
                sms = client.messages.create(
                    body=message,
                    from_=from_number,
                    to=phone,
                )
                logger.info(f"SMS alert sent to {phone} (SID: {sms.sid})")
            else:
                logger.warning("Twilio env vars not set — SMS skipped. "
                               "Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER.")
        except ImportError:
            logger.warning("twilio package not installed — SMS skipped. Run: pip install twilio")
        except Exception as e:
            logger.error(f"Failed to send SMS: {e}")


# ---------------------------------------------------------------------------
# Face identification (reused from Phase 3/4)
# ---------------------------------------------------------------------------
def identify_face(frame):
    """Returns (name, bbox) or ('unknown', None) or (None, None)."""
    if not os.path.isdir(DB_PATH) or not os.listdir(DB_PATH):
        return None, None
    try:
        results = DeepFace.find(
            img_path=frame,
            db_path=DB_PATH,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,
            silent=True,
        )
        if results and len(results[0]) > 0:
            best = results[0].iloc[0]
            identity_path = best["identity"]
            name = os.path.basename(os.path.dirname(identity_path))
            bbox = None
            sx = best.get("source_x")
            if sx is not None:
                bbox = (best["source_x"], best["source_y"],
                        best["source_w"], best["source_h"])
            return name, bbox
        return "unknown", None
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Stream connection with auto-reconnect
# ---------------------------------------------------------------------------
def open_stream(source):
    """Open a video source with retry logic for RTSP streams."""
    logger.info(f"Connecting to video source: {source}")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"Failed to open video source: {source}")
        return None
    logger.info("Video source connected successfully.")
    return cap


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="SentryCare Phase 5 — Production Prototype")
    parser.add_argument("--source", default="0",
                        help="Video source: 0 for webcam, or an RTSP URL (default: 0)")
    parser.add_argument("--headless", action="store_true",
                        help="Run without GUI window (for server/edge deployment)")
    parser.add_argument("--alert-phone", default=None,
                        help="Phone number to send SMS alerts to (requires Twilio setup)")
    parser.add_argument("--max-reconnect", type=int, default=5,
                        help="Max consecutive reconnect attempts for RTSP streams (default: 5)")
    args = parser.parse_args()

    # Parse source — integer means webcam index
    source = int(args.source) if args.source.isdigit() else args.source

    cap = open_stream(source)
    if cap is None:
        sys.exit(1)

    fsm = FallStateMachine()
    alerted = False
    frame_count = 0
    last_known_name = None
    last_bbox = None
    last_capture_time = 0
    capture_cooldown = 2.0
    reconnect_attempts = 0

    logger.info("SentryCare Phase 5 running. Press 'q' to quit.")

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ok, frame = cap.read()

            # --- Auto-reconnect logic for RTSP ---
            if not ok:
                if isinstance(source, str) and source.startswith("rtsp"):
                    reconnect_attempts += 1
                    if reconnect_attempts > args.max_reconnect:
                        logger.error(f"Max reconnect attempts ({args.max_reconnect}) exceeded. Exiting.")
                        break
                    logger.warning(f"Stream dropped. Reconnecting ({reconnect_attempts}/{args.max_reconnect})...")
                    cap.release()
                    time.sleep(2)
                    cap = open_stream(source)
                    if cap is None:
                        continue
                    continue
                else:
                    break

            reconnect_attempts = 0  # reset on successful read
            frame_count += 1
            h, w = frame.shape[:2]
            current_time = time.time()

            # --- Tier 1: Pose + fall signal (every frame) ---
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            down_now = False
            person_present = bool(results.pose_landmarks)
            if person_present:
                if not args.headless:
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                down_now = is_down_posture(results.pose_landmarks.landmark, h, w)

            state = fsm.update(down_now)

            # --- Tier 2: Face ID (throttled) ---
            if person_present and frame_count % FACE_ID_EVERY_N_FRAMES == 0:
                name, bbox = identify_face(frame)
                if name is not None:
                    last_known_name = name
                    last_bbox = bbox

                    # Capture unknown faces
                    if name == "unknown" and (current_time - last_capture_time) > capture_cooldown:
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        if bbox:
                            bx, by, bw, bh = bbox
                            face_roi = frame[by:by+bh, bx:bx+bw]
                            if face_roi.size > 0:
                                cv2.imwrite(os.path.join(OUTPUT_DIR, f"unknown_{timestamp}.jpg"), face_roi)
                        else:
                            cv2.imwrite(os.path.join(OUTPUT_DIR, f"unknown_{timestamp}.jpg"), frame)
                        logger.info(f"Unknown person captured at {timestamp}")
                        last_capture_time = current_time

            # --- Tier 3: Fall confirmed → Alert ---
            if state == CONFIRMED_FALL and not alerted:
                alerted = True
                who = last_known_name or "unidentified person"
                send_alert(who, "FALL", frame=frame, phone=args.alert_phone)
            elif state != CONFIRMED_FALL:
                alerted = False

            # --- GUI overlay (skip if headless) ---
            if not args.headless:
                color = {
                    "NORMAL": (0, 200, 0),
                    "POSSIBLE_FALL": (0, 165, 255),
                    "CONFIRMED_FALL": (0, 0, 255),
                }[state]
                cv2.putText(frame, f"STATE: {state}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                if last_known_name:
                    face_label = last_known_name if last_known_name != "unknown" else "Unknown"
                    face_color = (0, 255, 0) if last_known_name != "unknown" else (0, 0, 255)
                    cv2.putText(frame, f"ID: {face_label}", (20, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, face_color, 2)
                    if last_bbox and frame_count % FACE_ID_EVERY_N_FRAMES < 10:
                        bx, by, bw, bh = last_bbox
                        cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), face_color, 2)

                cv2.imshow("SentryCare — Phase 5", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cap.release()
    if not args.headless:
        cv2.destroyAllWindows()
    logger.info("SentryCare shutdown complete.")


if __name__ == "__main__":
    main()
