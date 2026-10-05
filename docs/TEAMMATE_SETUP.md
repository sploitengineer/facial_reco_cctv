# Teammate setup and test checklist

These instructions are for a fresh Windows checkout. The original developer's Python environment, downloaded models, photos, datasets, and camera configuration are local files; they are not included in GitHub. Each teammate must prepare their own machine.

## 1. Get the project

Open PowerShell and run:

```powershell
git clone https://github.com/sploitengineer/facial_reco_cctv.git
Set-Location facial_reco_cctv
```

If you already have a clean checkout, open its directory and run `git pull --ff-only` instead. Save any local work before updating.

**Why:** You need the current package, configuration examples, setup scripts, tests, and documentation.

## 2. Install the local environment

If `uv --version` works, run:

```powershell
.\scripts\setup.ps1
```

The script downloads Python 3.12 if needed, creates `.venv`, installs dependencies and SentryCare, copies the camera example to the ignored `configs/cameras.json` if missing, and checks inference imports. Downloads require internet access and several hundred MB of disk space. Initial model startup may take a little time.

If you prefer an existing Python 3.12 installation, or PowerShell blocks scripts, use:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
Copy-Item configs/cameras.example.json configs/cameras.json
.\.venv\Scripts\python.exe -m sentrycare.doctor
```

Use the manual `Copy-Item` only if you do not already have a camera configuration. If neither Python 3.12 nor uv is installed, install one using your usual approved software-installation process first. The uv installation guide is linked by `scripts/setup.ps1`.

**Why:** The current MediaPipe Pose implementation uses a legacy API and requires the documented Python/package versions. A system Python 3.13 environment is unsuitable for this inference setup.

**Expected:** Dependency, pose inference, and face-runtime checks pass. Data directories appear under `data/`. No camera or SMS is used by this initial check.

## 3. Check your webcam

Close other applications using the webcam, then run:

```powershell
.\.venv\Scripts\python.exe -m sentrycare.doctor --camera 0
```

**Why:** This verifies that your own camera can open and supply frames to MediaPipe. The developer's webcam test does not verify another machine.

**Expected:** `PASS webcam_inference`, reporting 30 processed frames. The report is saved to `data/logs/runtime_check.json`; this check does not save camera images. A zero pose count means no body pose was found, so check framing in the next step.

If the camera is unavailable, close video-call/camera apps and check Windows Settings > Privacy & security > Camera, including desktop-app access. If your intended camera uses another index, substitute it for `0`.

## 4. Test the live body tracking

```powershell
.\.venv\Scripts\python.exe -m sentrycare.app --source 0 --camera-id laptop --no-face-id
```

Stand far enough away that both shoulders and hips are visible. Try ordinary standing, sitting, and movement. Look for body landmarks and the displayed state. Press **q** to close the window.

**Why:** The posture heuristic uses shoulders and hips; a face-only close-up cannot provide the necessary body geometry. This step establishes a usable camera position.

Do not stage a dangerous fall. Current thresholds remain unvalidated, and the four-clip research smoke test missed a fall. A `NORMAL` state alone does not prove the system has detected a person; check for landmarks too.

## 5. Enroll yourself and test identification

Create a named folder:

```powershell
New-Item -ItemType Directory -Force data\known_faces\your_name
```

Place 3-5 clear reference photos in it, using different angles and lighting with one person per photo. Enroll other people only with their consent. The folder name becomes the displayed identity.

Run:

```powershell
.\.venv\Scripts\python.exe -m sentrycare.app --source 0 --camera-id laptop
```

Keep your face, shoulders, and hips visible. Recognition currently runs only when a body pose is present. Initial SFace use downloads its weights into `data/models/`.

**Why:** The pre-trained model needs local reference photos to match you. Without them, detected faces are classified as unknown. This verifies actual identification, which the synthetic model check does not establish.

**Expected:** Your folder name appears as the identity when recognized. Record missed matches or incorrect names. Unknown-face snapshots can appear in `data/captured_faces/laptop/`. Stop with **q**.

## 6. Test the background camera supervisor

Close the live-view window first, then run:

```powershell
.\.venv\Scripts\python.exe -m sentrycare.cameras --config configs/cameras.json
```

There is no video window. In a second terminal in the same project directory:

```powershell
Get-Content data\logs\cameras_status.json
Get-Content data\logs\heartbeat_laptop.json
```

**Why:** This verifies worker startup, progress reporting, and process supervision before adding CCTV streams. The example enables only the laptop camera and disables face identification for that background worker.

**Expected:** Camera status reaches `processing` and the heartbeat's frame count increases. Stop with **Ctrl+C**. Do not run the live viewer and supervisor against the same webcam at the same time.

## 7. Run the regression tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

**Why:** This checks confirmation timing, evaluation separation, event logging, stream recovery, sequence timestamps, and supervisor behavior without opening a camera or sending notifications.

**Expected:** All 29 current tests pass. New tests may increase this count in later commits.

## 8. Optional: reproduce the research smoke test

```powershell
.\.venv\Scripts\python.exe scripts/download_urfd_samples.py
.\.venv\Scripts\python.exe -m sentrycare.evaluation extract data/evaluation/urfd_samples.csv --output data/evaluation/urfd_features.json
.\.venv\Scripts\python.exe -m sentrycare.evaluation tune data/evaluation/urfd_features.json --report data/evaluation/urfd_report.json --config-output data/evaluation/urfd_candidate.json
```

**Why:** This reproduces the full download, timestamped pose extraction, threshold search, and reporting pipeline. The official URFD source permits non-commercial academic use under CC BY-NC-SA 4.0; the downloader retains attribution. See `VALIDATION.md` for the original results.

The four recordings are a pipeline sample, not an accuracy benchmark. The initial training fall was missed across the default grid. Do not apply the exported candidate to a deployment. Next work is analysis of that miss and a broader subject/session-separated evaluation.

## What to share with the team

Report your Python version, webcam index, whether landmarks appeared, whether your enrolled name was recognized, supervisor status/frame progress, and any errors. Share test results or relevant error text rather than personal photos, tokens, or camera credentials.

The next optional integrations are a real RTSP camera and Twilio notifications. They require your own camera address/account configuration; no signup or SMS is needed for the steps above. See `NEXT_STEPS.md` and the main README for those integrations.
