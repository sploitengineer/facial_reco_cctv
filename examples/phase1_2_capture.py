import cv2
import os
import time

# Create a directory to store captured faces
from sentrycare.paths import CAPTURED_FACES
OUTPUT_DIR = str(CAPTURED_FACES)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Use OpenCV's built-in Haar cascade for fast, lightweight face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def main():
    cap = cv2.VideoCapture(0)
    print(f"Starting Phase 1 & 2: Face Detection & Capture.")
    print(f"Capturing to '{OUTPUT_DIR}' directory. Press 'q' to quit.")

    # We'll limit capture rate to avoid spamming images
    last_capture_time = 0
    capture_cooldown = 1.0  # seconds

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale for Haar cascade
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        current_time = time.time()
        for (x, y, w, h) in faces:
            # Draw bounding box (Phase 1)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, "Face Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Capture and timestamp (Phase 2)
            if current_time - last_capture_time > capture_cooldown:
                # Extract the region of interest
                face_roi = frame[y:y+h, x:x+w]

                if face_roi.size > 0:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"face_{timestamp}.jpg"
                    filepath = os.path.join(OUTPUT_DIR, filename)
                    cv2.imwrite(filepath, face_roi)
                    print(f"Captured {filename}")
                    last_capture_time = current_time

        cv2.imshow("Phase 1 & 2: Face Detection & Capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
