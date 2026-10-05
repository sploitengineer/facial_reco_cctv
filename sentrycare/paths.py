"""Project paths independent of the caller's working directory."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("SENTRYCARE_DATA_DIR", PROJECT_ROOT / "data")).resolve()
KNOWN_FACES = DATA_DIR / "known_faces"
CAPTURED_FACES = DATA_DIR / "captured_faces"
EVENT_LOGS = DATA_DIR / "event_logs"
LOGS = DATA_DIR / "logs"
# Keep downloaded face models in the project's ignored data directory.
os.environ.setdefault("DEEPFACE_HOME", str(DATA_DIR / "models"))


def ensure_data_dirs():
    for name in ("known_faces", "captured_faces", "event_logs", "logs", "datasets", "evaluation"):
        (DATA_DIR / name).mkdir(parents=True, exist_ok=True)
