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
from collections import deque

import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# --- Tunable thresholds — validate these against UR Fall / Le2i before trusting them ---
ASPECT_RATIO_THRESHOLD = 1.3   # body bounding box wider than tall => lying-down candidate
LOW_POSITION_THRESHOLD = 0.55  # torso below this fraction of frame height => "on the ground" zone
CONFIRM_SECONDS = 3.0          # must stay in "down" state this long before we call it a real fall
RECOVER_FRAMES = 8             # consecutive "not down" frames needed to cancel a possible fall

# States
NORMAL, POSSIBLE_FALL, CONFIRMED_FALL = "NORMAL", "POSSIBLE_FALL", "CONFIRMED_FALL"


def is_down_posture(landmarks, frame_h, frame_w):
    """Cheap geometric heuristic: wide+low bounding box of shoulders/hips = likely down."""
    pts = [
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER],
        landmarks[mp_pose.PoseLandmark.LEFT_HIP],
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP],
    ]
    xs = [p.x for p in pts]
    ys = [p.y for p in pts]
    width = (max(xs) - min(xs)) * frame_w
    height = (max(ys) - min(ys)) * frame_h + 1e-6
    aspect_ratio = width / height

    torso_y = sum(ys) / len(ys)  # 0 = top of frame, 1 = bottom

    return aspect_ratio > ASPECT_RATIO_THRESHOLD and torso_y > LOW_POSITION_THRESHOLD


class FallStateMachine:
    def __init__(self):
        self.state = NORMAL
        self.down_since = None
        self.recover_streak = 0

    def update(self, down_now: bool) -> str:
        if self.state == NORMAL:
            if down_now:
                self.state = POSSIBLE_FALL
                self.down_since = time.time()
                self.recover_streak = 0

        elif self.state == POSSIBLE_FALL:
            if down_now:
                self.recover_streak = 0
                if time.time() - self.down_since >= CONFIRM_SECONDS:
                    self.state = CONFIRMED_FALL
            else:
                self.recover_streak += 1
                if self.recover_streak >= RECOVER_FRAMES:
                    # got back up before confirmation — false alarm avoided
                    self.state = NORMAL
                    self.down_since = None

        elif self.state == CONFIRMED_FALL:
            if not down_now:
                self.recover_streak += 1
                if self.recover_streak >= RECOVER_FRAMES:
                    self.state = NORMAL
                    self.down_since = None
            else:
                self.recover_streak = 0

        return self.state


def main():
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
