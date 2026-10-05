"""
Step 1 prototype — combined loop: pose/fall-detection every frame,
face recognition throttled to every FACE_ID_EVERY_N_FRAMES frames.

This mirrors the tiered architecture from the design doc at small scale:
- Pose runs continuously (cheap) — this is your Tier 1/2 presence + fall signal.
- Face ID runs on a slower cadence (more expensive per call) — this stands
  in for "only run the expensive check when it's worth it."
- A CONFIRMED_FALL print statement stands in for the Tier 3 GPU burst +
  WhatsApp/SMS alert you'll wire up later.

Run:
  python combined_prototype.py
Press 'q' to quit.
"""

import os
import time
from collections import deque

import cv2
import mediapipe as mp
from sentrycare.face_identity import identify_face as recognize_face

from sentrycare.fall_detection import FallStateMachine, is_down_posture, CONFIRMED_FALL

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

from sentrycare.paths import KNOWN_FACES, CAPTURED_FACES
DB_PATH = str(KNOWN_FACES)
MODEL_NAME = "SFace"
DETECTOR_BACKEND = "opencv"
FACE_ID_EVERY_N_FRAMES = 15  # ~every 0.5s at 30fps — tune based on your machine's speed


def identify_face(frame):
    return recognize_face(frame, DB_PATH, MODEL_NAME, DETECTOR_BACKEND)[0]

def main():
    cap = cv2.VideoCapture(0)
    fsm = FallStateMachine()
    alerted = False
    frame_count = 0
    last_known_name = None

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            frame_count += 1
            h, w = frame.shape[:2]

            # --- Tier 1/2: pose + fall signal, every frame ---
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            down_now = False
            person_present = bool(results.pose_landmarks)
            if person_present:
                down_now = is_down_posture(results.pose_landmarks.landmark, h, w)

            state = fsm.update(down_now)
            if not person_present:
                last_known_name = None

            # --- Face ID: only when a person is present, throttled ---
            if person_present and frame_count % FACE_ID_EVERY_N_FRAMES == 0:
                name = identify_face(frame)
                if name is not None:
                    last_known_name = name

            # --- Tier 3 handoff point ---
            if state == CONFIRMED_FALL and not alerted:
                alerted = True
                who = last_known_name or "unidentified person"
                print(f"[{time.strftime('%H:%M:%S')}] FALL CONFIRMED for '{who}' "
                      f"— would trigger GPU confirmation + WhatsApp/SMS alert now")
            elif state != CONFIRMED_FALL:
                alerted = False

            # --- Overlay ---
            if person_present:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            color = {"NORMAL": (0, 200, 0), "POSSIBLE_FALL": (0, 165, 255), "CONFIRMED_FALL": (0, 0, 255)}[state]
            cv2.putText(frame, f"STATE: {state}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            if last_known_name:
                cv2.putText(frame, f"ID: {last_known_name}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            cv2.imshow("Combined Prototype (Face ID + Fall Detection)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
