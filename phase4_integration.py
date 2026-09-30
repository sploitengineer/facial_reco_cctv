import os
import time
import cv2
import mediapipe as mp
from deepface import DeepFace

from fall_detection_webcam import FallStateMachine, is_down_posture, CONFIRMED_FALL

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

DB_PATH = "known_faces"
OUTPUT_DIR = "captured_faces"
MODEL_NAME = "SFace"
DETECTOR_BACKEND = "opencv"
FACE_ID_EVERY_N_FRAMES = 15

os.makedirs(DB_PATH, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def identify_face(frame):
    """Returns the best-matched name and bbox, or 'unknown' / None if nothing usable found."""
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
            
            source_x = best.get("source_x")
            source_y = best.get("source_y")
            source_w = best.get("source_w")
            source_h = best.get("source_h")
            bbox = None
            if source_x is not None:
                bbox = (source_x, source_y, source_w, source_h)
            return name, bbox
        return "unknown", None
    except Exception:
        return None, None

def main():
    cap = cv2.VideoCapture(0)
    fsm = FallStateMachine()
    alerted = False
    frame_count = 0
    last_known_name = None
    last_bbox = None
    
    last_capture_time = 0
    capture_cooldown = 2.0

    print("Starting Phase 4: Integration (Pose + Face Recognition + Unknown Capture)")
    print("Press 'q' to quit.")
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            frame_count += 1
            h, w = frame.shape[:2]
            current_time = time.time()

            # --- Tier 1/2: pose + fall signal, every frame ---
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            down_now = False
            person_present = bool(results.pose_landmarks)
            if person_present:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                down_now = is_down_posture(results.pose_landmarks.landmark, h, w)

            state = fsm.update(down_now)

            # --- Face ID: only when a person is present, throttled ---
            if person_present and frame_count % FACE_ID_EVERY_N_FRAMES == 0:
                name, bbox = identify_face(frame)
                if name is not None:
                    last_known_name = name
                    last_bbox = bbox
                    
                    # Phase 3 integration: Capture unknown faces with timestamp
                    if name == "unknown" and (current_time - last_capture_time) > capture_cooldown:
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        if bbox:
                            bx, by, bw, bh = bbox
                            face_roi = frame[by:by+bh, bx:bx+bw]
                            if face_roi.size > 0:
                                cv2.imwrite(os.path.join(OUTPUT_DIR, f"unknown_{timestamp}.jpg"), face_roi)
                        else:
                            cv2.imwrite(os.path.join(OUTPUT_DIR, f"unknown_{timestamp}.jpg"), frame)
                        print(f"Captured unknown person at {timestamp}")
                        last_capture_time = current_time

            # --- Tier 3 handoff point (Fall Detected) ---
            if state == CONFIRMED_FALL and not alerted:
                alerted = True
                who = last_known_name or "unidentified person"
                print(f"[{time.strftime('%H:%M:%S')}] FALL CONFIRMED for '{who}' "
                      f"— logging event and triggering alert protocol!")
            elif state != CONFIRMED_FALL:
                alerted = False

            # --- Overlay ---
            color = {"NORMAL": (0, 200, 0), "POSSIBLE_FALL": (0, 165, 255), "CONFIRMED_FALL": (0, 0, 255)}[state]
            cv2.putText(frame, f"STATE: {state}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            if last_known_name:
                face_label = last_known_name if last_known_name != "unknown" else "Unknown"
                face_color = (0, 255, 0) if last_known_name != "unknown" else (0, 0, 255)
                cv2.putText(frame, f"ID: {face_label}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, face_color, 2)
                
                # Draw bounding box if we have one and we just updated it recently
                if last_bbox and frame_count % FACE_ID_EVERY_N_FRAMES < 10:
                    bx, by, bw, bh = last_bbox
                    cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), face_color, 2)

            cv2.imshow("Phase 4: Fall Detection Integration", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
