"""
Step 1 prototype — pose-based fall detection from a laptop webcam.

Uses MediaPipe Pose to extract body landmarks every frame, derives a simple
"is this person in a fallen posture right now?" signal from geometry (torso
orientation + vertical position), then runs that signal through a small
state machine so a single ambiguous frame never fires an alert on its own —
matching the temporal-confirmation approach used in published CPU-only
fall-detection systems for elder care.

This is deliberately simple and rule-based, NOT a trained classifier. Once
you have UR Fall Detection Dataset / Le2i clips, run this same landmark
extraction over those labeled videos to tune the thresholds below before
trusting it on your own webcam.

Run:
  python fall_detection_webcam.py
Press 'q' to quit.
"""

import time
from sentrycare.fall_detection import (
    FallStateMachine, is_down_posture, NORMAL, POSSIBLE_FALL, CONFIRMED_FALL,
    DEFAULT_CONFIG,
)

# --- Tunable thresholds — validate these against UR Fall / Le2i before trusting them ---
ASPECT_RATIO_THRESHOLD = DEFAULT_CONFIG.aspect_ratio_threshold
LOW_POSITION_THRESHOLD = DEFAULT_CONFIG.low_position_threshold
CONFIRM_SECONDS = DEFAULT_CONFIG.confirm_seconds
RECOVER_FRAMES = DEFAULT_CONFIG.recover_frames

# States
def main():
    import cv2
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)
    fsm = FallStateMachine()
    alerted = False

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            down_now = False
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                down_now = is_down_posture(results.pose_landmarks.landmark, h, w)

            state = fsm.update(down_now)

            if state == CONFIRMED_FALL and not alerted:
                alerted = True
                # --- This is the Tier 2 -> Tier 3 handoff point ---
                # In the full architecture this publishes an event and
                # triggers the GPU confirmation burst + WhatsApp/SMS alert.
                # For this local prototype, we just print it.
                print(f"[{time.strftime('%H:%M:%S')}] FALL CONFIRMED — would trigger alert now")
            elif state != CONFIRMED_FALL:
                alerted = False

            color = {"NORMAL": (0, 200, 0), "POSSIBLE_FALL": (0, 165, 255), "CONFIRMED_FALL": (0, 0, 255)}[state]
            cv2.putText(frame, f"STATE: {state}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.imshow("Fall Detection Prototype", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
