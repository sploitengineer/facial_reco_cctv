import cv2
import os
import time
from sentrycare.face_identity import identify_face as recognize_face

# Configuration
from sentrycare.paths import KNOWN_FACES, CAPTURED_FACES
DB_PATH = str(KNOWN_FACES)
from sentrycare.paths import CAPTURED_FACES
OUTPUT_DIR = str(CAPTURED_FACES)
MODEL_NAME = "SFace"
DETECTOR_BACKEND = "opencv"

os.makedirs(DB_PATH, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def identify_face(frame):
    return recognize_face(frame, DB_PATH, MODEL_NAME, DETECTOR_BACKEND)

def main():
    cap = cv2.VideoCapture(0)
    print(f"Starting Phase 3: Face Recognition.")
    print(f"Identifying against '{DB_PATH}' folder. Unknown faces will be saved to '{OUTPUT_DIR}'.")
    print(f"Press 'q' to quit.")

    last_capture_time = 0
    capture_cooldown = 2.0  # limit capture of unknown faces

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()

        # DeepFace.find is slow. In a production app (Phase 5), we'd run detection
        # on every frame and recognition on every Nth frame.
        # For Phase 3 local testing, we'll just run it per frame for simplicity.
        name, bbox = identify_face(frame)

        if name:
            label = name if name != "unknown" else "Unknown"
            color = (0, 255, 0) if name != "unknown" else (0, 0, 255)

            if bbox:
                x, y, w, h = bbox
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            else:
                # Fallback if no bbox returned but recognized
                cv2.putText(frame, f"Detected: {label}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # Phase 2 logic applied: Capture unknown faces
            if name == "unknown" and (current_time - last_capture_time) > capture_cooldown:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                if bbox:
                    x, y, w, h = bbox
                    face_roi = frame[y:y+h, x:x+w]
                    if face_roi.size > 0:
                        cv2.imwrite(os.path.join(OUTPUT_DIR, f"unknown_{timestamp}.jpg"), face_roi)
                else:
                    cv2.imwrite(os.path.join(OUTPUT_DIR, f"unknown_{timestamp}.jpg"), frame)

                print(f"Captured unknown person at {timestamp}")
                last_capture_time = current_time

        cv2.imshow("Phase 3: Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
