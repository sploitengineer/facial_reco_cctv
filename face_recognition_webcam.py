"""
Step 1 prototype — face recognition from a laptop webcam.

Enrollment:
  Put 3-5 clear reference photos of each person into:
      known_faces/<person_name>/photo1.jpg
      known_faces/<person_name>/photo2.jpg
      ...
  e.g. known_faces/kaushal/1.jpg, known_faces/kaushal/2.jpg

Run:
  python face_recognition_webcam.py

This uses DeepFace's built-in webcam stream, which handles detection,
alignment, embedding, and matching against the known_faces/ folder for you.
It's the fastest way to see face recognition working end to end before
we build anything custom.

Model choice matters for the production build:
- "Buffalo_L" / "SFace" / "GhostFaceNet" are lighter, edge-friendly options
- "ArcFace" is heavier but more accurate — good for a Tier-3 confirmation step
Swap MODEL_NAME below to compare.
"""

from deepface import DeepFace

DB_PATH = "known_faces"
MODEL_NAME = "SFace"          # lightweight — good for a continuous Tier-2 style loop
DETECTOR_BACKEND = "opencv"   # fastest detector; swap to "retinaface" for higher accuracy, slower

if __name__ == "__main__":
    print(f"Starting webcam face recognition against '{DB_PATH}' "
          f"(model={MODEL_NAME}, detector={DETECTOR_BACKEND})")
    print("Press 'q' in the video window to quit.")

    # DeepFace.stream() opens the webcam, detects a face held steady for a
    # few frames, then shows the matched identity (or "unknown") for a
    # few seconds — good enough to validate accuracy before wiring anything
    # else up.
    DeepFace.stream(
        db_path=DB_PATH,
        model_name=MODEL_NAME,
        detector_backend=DETECTOR_BACKEND,
        enable_face_analysis=False,  # turn off age/gender/emotion, we don't need it
    )
