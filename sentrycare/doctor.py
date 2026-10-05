"""Check installed inference libraries, optional SFace weights, and webcam frames."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import time

from .paths import LOGS, ensure_data_dirs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, help="Read this webcam without saving images")
    parser.add_argument("--face-model", action="store_true", help="Download/load SFace and verify an embedding")
    args = parser.parse_args(argv)
    ensure_data_dirs()
    report = {"python": platform.python_version(), "checks": {}}
    def check(name, operation):
        try:
            report["checks"][name] = {"ok": True, "details": operation()}
            print(f"PASS {name}: {report['checks'][name]['details']}")
        except Exception as exc:
            report["checks"][name] = {"ok": False, "error": str(exc)}
            print(f"FAIL {name}: {exc}")

    def versions():
        return {name: importlib.metadata.version(name) for name in ("opencv-python", "mediapipe", "deepface", "tensorflow", "numpy")}
    check("dependencies", versions)

    def pose_test():
        import cv2
        import mediapipe as mp
        import numpy as np
        with mp.solutions.pose.Pose() as pose:
            result = pose.process(cv2.cvtColor(np.zeros((480, 640, 3), dtype=np.uint8), cv2.COLOR_BGR2RGB))
        if result.pose_landmarks is not None:
            raise RuntimeError("Blank frame unexpectedly returned a pose")
        return "Pose model initialized and processed a blank frame"
    check("pose_inference", pose_test)

    def face_import():
        from deepface import DeepFace
        return "DeepFace imports successfully"
    check("face_runtime", face_import)
    if args.face_model:
        def face_test():
            from deepface import DeepFace
            import numpy as np
            result = DeepFace.represent(np.zeros((112, 112, 3), dtype=np.uint8), model_name="SFace",
                                        detector_backend="skip", enforce_detection=False)
            if len(result[0]["embedding"]) != 128:
                raise RuntimeError("Unexpected SFace embedding dimensions")
            return "SFace weights loaded; synthetic input produced a 128-dimensional embedding"
        check("sface_model", face_test)
    if args.camera is not None:
        def camera_test():
            import cv2
            import mediapipe as mp
            cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY)
            count, poses = 0, 0
            started = time.monotonic()
            try:
                if not cap.isOpened():
                    raise RuntimeError("Webcam unavailable; check Windows camera access and other camera apps")
                with mp.solutions.pose.Pose() as pose:
                    while count < 30:
                        ok, frame = cap.read()
                        if not ok:
                            raise RuntimeError("Webcam opened but could not supply frames")
                        result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        count += 1
                        poses += int(result.pose_landmarks is not None)
                return {"frames_processed": count, "frames_with_pose": poses,
                        "elapsed_seconds": round(time.monotonic() - started, 2)}
            finally:
                cap.release()
        check("webcam_inference", camera_test)
    path = LOGS / "runtime_check.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {path}")
    return 0 if all(check["ok"] for check in report["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
