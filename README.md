# SentryCare

CPU-based CCTV analytics for aged-care resident safety: pose-based fall events, detected-face identification, RTSP reconnects, event snapshots, and optional Twilio notifications.

This is an academic prototype. Fall detection uses geometry and temporal confirmation; GPU confirmation, intrusion alert policy, multi-person tracking, dashboard, enrollment UI, and cloud deployment remain future work.

## Directory structure

```text
facial_recognition/
├── sentrycare/              # Application package
│   ├── app.py               # One camera: inference, snapshots, alerts, heartbeat
│   ├── cameras.py           # Multiple worker processes, watchdog, restarts
│   ├── doctor.py            # Runtime, model, and webcam checks
│   ├── evaluation.py        # Feature extraction, threshold search, reports
│   ├── recordings.py        # Video and timestamped RGB-sequence readers
│   ├── fall_detection.py    # Posture features and temporal state machine
│   ├── face_identity.py     # Detected-face matching using SFace
│   ├── video_stream.py      # Video input, timeouts, reconnect, cleanup
│   └── paths.py             # Shared data paths and local model cache
├── examples/                # Earlier webcam prototypes
├── configs/                 # Default fall settings and sample camera configuration
├── scripts/                 # Setup, launch, and official URFD sample downloader
├── tests/                   # Dependency-free regression tests
├── docs/                    # Proposal, blueprints, validation notes, next steps
├── evaluation/              # Generic example manifest
├── data/                    # Ignored local runtime and research data
│   ├── known_faces/         # One folder per enrolled person
│   ├── captured_faces/      # Unknown faces, separated by camera ID
│   ├── event_logs/          # Event CSVs and snapshots, separated by camera ID
│   ├── logs/                # Logs, health report, heartbeat, supervisor status
│   ├── models/              # Downloaded SFace weights
│   ├── datasets/            # RGB recordings, timing files, attribution
│   └── evaluation/          # Manifests, cached features, reports, exported settings
├── .runtime/                # Local Python 3.12 (ignored)
├── .venv/                   # Installed dependencies (ignored)
├── pyproject.toml           # Package and console commands
└── phase*.py                # Compatibility launchers for earlier commands
```

The proposal is now at `docs/SentryCare_Project_Proposal.pdf`. Existing captured photos were moved into `data/captured_faces/` without modification. Old commands such as `python phase5_production.py` still launch the packaged implementation.

## Run on this machine

The project-local Python 3.12 environment and dependencies are installed. From the project directory:

```powershell
# Webcam with pose detection; press q to stop
.\scripts\run.ps1

# Webcam with face identification as well
.\scripts\run.ps1 -FaceId

# Check webcam and installed inference libraries
.\scripts\run.ps1 -Mode doctor

# Headless configured camera workers; Ctrl+C stops the supervisor
.\scripts\run.ps1 -Mode cameras
```

If PowerShell blocks local scripts, use the Python commands directly:

```powershell
.venv\Scripts\python -m sentrycare.app --source 0 --camera-id laptop --no-face-id
.venv\Scripts\python -m sentrycare.doctor --camera 0
.venv\Scripts\python -m sentrycare.cameras --config configs/cameras.json
```

`--headless` removes the video window. `--max-frames 30` limits a worker smoke test. Unknown faces are saved only when face identification is enabled and a face was actually detected.

## Set up a fresh checkout

For a step-by-step guide explaining what to run, what to expect, and why, see [Teammate setup](docs/TEAMMATE_SETUP.md).

Use Python 3.11/3.12. The legacy `mp.solutions.pose` implementation requires MediaPipe 0.10.21 ([release files](https://pypi.org/project/mediapipe/0.10.21/#files)).

```powershell
# With uv already installed; downloads local Python if needed
.\scripts\setup.ps1

# Or use an installed Python 3.12
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install --no-deps -e .
Copy-Item configs/cameras.example.json configs/cameras.json
```

On Linux/macOS use `python3.12 -m venv .venv`, `.venv/bin/python`, and the module commands. The Windows setup scripts and live webcam path have been exercised on this machine. Keep the virtual environment local to this checkout.

## Camera configuration

Edit the ignored `configs/cameras.json`. Camera `laptop` is enabled with source `0` and face identification disabled for the first live run. Enable face ID by setting `face_id` to `true`. Each enabled camera must have a unique ID and video source.

The sample hallway camera uses an environment variable so its credentials stay outside tracked JSON:

```powershell
$env:SENTRYCARE_HALLWAY_RTSP = 'rtsp://user:password@camera-address:554/stream'
.venv\Scripts\python -m sentrycare.cameras --config configs/cameras.json --check
```

Each worker runs in its own process and has its own pose tracker, fall state, snapshots, and logs. RTSP opens/reads have timeouts; the supervisor restarts failed or stalled workers within configured limits. Watch progress in `data/logs/cameras_status.json` and `heartbeat_<id>.json`. All workers stop when the supervisor exits. `--duration 25` limits a supervisor smoke test. Simultaneous inference across several real cameras still needs hardware profiling.

## Face enrollment and notifications

Place 3-5 clear photos per consenting person under `data/known_faces/<person_name>/`. The directory name becomes the displayed identity. Enrollment and image data are ignored by Git. DeepFace's model cache lives under `data/models/`; it can download weights on first use.

Notifications are optional. Install Twilio, set credentials in your shell, and supply a destination explicitly:

```powershell
.venv\Scripts\python -m pip install twilio
$env:TWILIO_ACCOUNT_SID = 'your-account-sid'
$env:TWILIO_AUTH_TOKEN = 'your-auth-token'
$env:TWILIO_FROM_NUMBER = 'your-sender-number'
.venv\Scripts\python -m sentrycare.app --alert-phone '+your-destination-number'
```

`.env.example` documents variables; `.env` is not loaded automatically. No SMS was sent during verification. CSV delivery statuses distinguish console logging, missing setup, failures, and Twilio acceptance; acceptance is not proof of handset delivery.

## Fall evaluation

The four-clip sample downloader uses the official [URFD dataset](https://fenix.ur.edu.pl/~mkepski/ds/uf.html). It downloads camera-0 RGB sequences and millisecond timing CSVs, preserves attribution, and creates a manifest. The source permits non-commercial academic use under CC BY-NC-SA 4.0. This small manifest is a pipeline smoke test, not subject-disjoint accuracy validation.

```powershell
.venv\Scripts\python scripts/download_urfd_samples.py
.venv\Scripts\python -m sentrycare.evaluation extract data/evaluation/urfd_samples.csv --output data/evaluation/urfd_features.json
.venv\Scripts\python -m sentrycare.evaluation tune data/evaluation/urfd_features.json --report data/evaluation/urfd_report.json --config-output data/evaluation/urfd_candidate.json
```

For full evaluation, copy `evaluation/manifest.example.csv` and list your own recordings. Paths are relative to the manifest. Labels are `fall`/`adl`; splits are `train`/`validation`/`test`. Training requires both labels. Video files use their recorded FPS; URFD camera-0 image directories require a `timestamps` column pointing to the timing CSV. Split by subject/camera/session and reserve a final test set.

Feature extraction runs MediaPipe once. Threshold search uses only the standard library, tests 27 combinations by default, and chooses by training F1 with fewer false positives as the tie-breaker. Held-out clips do not affect selection. Reports contain clip-level confusion counts, precision, recall, specificity, F1, and first-alert timestamps. These timestamps are relative to recording start, not fall-onset latency.

Default settings in `configs/fall.default.json` remain unchanged after sample evaluation. Load an appropriately validated configuration using `--fall-config <path>`. The current heuristic can confuse deliberate lying down with a fall and tracks one pose per camera.

## Verification and remaining work

```powershell
.venv\Scripts\python -m unittest discover -s tests -v
uv pip check --python .venv/Scripts/python.exe --cache-dir tmp/uv-cache
```

Tests cover confirmation/recovery, unreliable poses, recording timestamps, train/held-out separation, face detection without enrollment, event logging, reconnects, worker configuration, restart limits, watchdogs, and cleanup. See `docs/VALIDATION.md` for actual integration results and `docs/NEXT_STEPS.md` for the steps requiring your input.

Next development areas: full dataset evaluation, target-hardware profiling, searchable event storage, live dashboard, enrollment UI, multi-person tracking, and the proposal's GPU confirmation tier.

## Team

Kaushalendra Pratap Singh (AU2644013), Dhruv Jain (AU2644029), Nandish Bhatt (AU2644012), Dev Upadhyay (AU2644003).
