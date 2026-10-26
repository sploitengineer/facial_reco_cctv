"""Guard against accidentally tracking local photographs, biometrics, or secrets."""

from pathlib import Path
import re
import subprocess
import unittest


ROOT = Path(__file__).resolve().parent.parent
PRIVATE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".heic", ".heif",
                    ".mp4", ".avi", ".mov", ".mkv", ".webm", ".y4m", ".yuv", ".zip", ".npy", ".npz",
                    ".pkl", ".pickle", ".joblib", ".db", ".sqlite", ".sqlite3", ".onnx", ".tflite",
                    ".h5", ".keras", ".pt", ".pth", ".pem", ".key", ".p12"}


def git(*args, input=None):
    return subprocess.run(["git", *args], cwd=ROOT, input=input, text=True, capture_output=True, check=False)


class RepositoryPrivacyTests(unittest.TestCase):
    def test_no_private_files_are_tracked(self):
        result = git("ls-files", "--cached")
        self.assertEqual(result.returncode, 0, result.stderr)
        private = []
        for name in result.stdout.splitlines():
            path = Path(name)
            if (path.suffix.lower() in PRIVATE_SUFFIXES
                    or (name.startswith("data/") and name != "data/README.md")
                    or name.startswith(("tmp/", ".venv/", ".runtime/", ".aws/"))
                    or re.search(r"\.(sqlite3?|db)-(wal|shm|journal)$", name)
                    or (path.name.startswith(".env") and path.name != ".env.example")
                    or (name.startswith("configs/cameras") and name != "configs/cameras.example.json")):
                private.append(name)
        self.assertEqual(private, [], f"Private files must not be tracked: {private}")

    def test_sensitive_paths_are_ignored_anywhere(self):
        names = ["portrait.jpg", "photo.JPEG", "exports/person.png", "capture.webm", "features.npy",
                 "features.npz", "faces.pkl", "registry.sqlite3", "registry.sqlite3-wal", ".env.local",
                 "configs/cameras.json", "configs/cameras.office.json", "data/studio/enrollment.sqlite3",
                 "data/studio/images/portrait.jpg", "tmp/private_diagnostics.json", "local_model.onnx",
                 "account.pem", "facial_reco_cctv.zip"]
        result = git("check-ignore", "--no-index", "-z", "--stdin", input="\0".join(names) + "\0")
        self.assertEqual(set(result.stdout.rstrip("\0").split("\0")), set(names))

    def test_public_setup_files_remain_shareable(self):
        result = git("check-ignore", "--no-index", "-z", "--stdin", input=".env.example\0configs/cameras.example.json\0data/README.md\0")
        self.assertEqual(result.stdout, "")
