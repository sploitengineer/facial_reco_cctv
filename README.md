# SentryCare — AI-Powered CCTV Analytics for Aged Care

Real-time facial recognition + fall detection over existing CCTV cameras (RTSP streams), built for aged-care safety.

---

## Progress Tracker

| Phase | Feature | Status |
|-------|---------|--------|
| 1 | Face Detection (Bounding Box) | ✅ Done |
| 2 | Face Capture with Timestamp | ✅ Done |
| 3 | Facial Recognition (Known vs Unknown) | ✅ Done |
| 4 | Fall Detection Integration (Pose + Face ID) | ✅ Done |
| 5 | RTSP Stream + Alerting (SMS/WhatsApp) | ✅ Done |
| 6 | Fall Threshold Tuning (URFD/Le2i datasets) | 🔧 In Progress |
| 7 | Multi-Camera Stream Manager | 🔲 Planned |
| 8 | Web Dashboard (Live View + Event History) | 🔲 Planned |
| 9 | Face Enrollment Web UI | 🔲 Planned |
| 10 | GCP Cloud Deployment | 🔲 Planned |
| 11 | Liveness / Anti-Spoofing Detection | 🔲 Planned |

**Currently working on:** Tuning fall detection thresholds against real datasets (URFD / Le2i).

---

## Roadmap & Next Steps

### Immediate (Current Sprint)

- [ ] **Threshold Tuning:** Run `is_down_posture()` landmark extraction over labeled URFD and Le2i video clips to find optimal values for `ASPECT_RATIO_THRESHOLD`, `LOW_POSITION_THRESHOLD`, and `CONFIRM_SECONDS` — the current values are starting guesses, not validated.
- [ ] **Model Benchmarking:** Compare SFace vs ArcFace vs GhostFaceNet accuracy and CPU cost per frame to pick the best model for always-on edge deployment.
- [ ] **Edge Hardware Profiling:** Benchmark actual FPS and latency on the target deployment hardware (laptop GPU, Jetson, or cloud VM).

### Short-Term (Next 2-4 Weeks)

- [ ] **Multi-Camera Stream Manager:** Build a process manager that handles N RTSP streams in parallel, each with its own inference thread and watchdog for reconnection.
- [ ] **Web Dashboard:** A simple Flask/FastAPI + HTML dashboard showing:
  - Live camera grid view
  - Event history table (falls, unknown detections) with snapshot thumbnails
  - Per-camera status indicators (online/offline/alert)
- [ ] **Face Enrollment UI:** A web page to upload photos and register new residents/staff into `known_faces/` without touching the filesystem manually.
- [ ] **Database Integration:** Move from CSV event logs to a proper database (PostgreSQL or SQLite) for searchable, filterable event history.
- [ ] **WhatsApp Business API:** Replace Twilio SMS with WhatsApp Business Cloud API for richer alerts (photo of the event + location tag).

### Long-Term (Production Readiness)

- [ ] **GCP Cloud Deployment:** Deploy the tiered architecture on Google Cloud:
  - Tier 1/2 on Compute Engine (always-on, CPU-only)
  - Tier 3 on Cloud Run with GPU (burst only on confirmed events)
  - Pub/Sub for event routing between tiers
- [ ] **Liveness / Anti-Spoofing:** Integrate passive liveness detection (e.g., MiniFASNet) to prevent photo-based spoofing of the face recognition system.
- [ ] **Trained Fall Classifier:** Move beyond geometric heuristics to a trained classifier on pose keypoint sequences (e.g., Temporal CNN or LSTM on URFD/Le2i features) for higher accuracy.
- [ ] **Multi-Person Tracking:** Track and identify multiple people simultaneously in the same frame using person re-identification.
- [ ] **Night Vision Handling:** Add IR/low-light preprocessing pipeline for cameras with poor night-time image quality.
- [ ] **Audit & Compliance:** Implement data retention policies, access controls, and anonymization for biometric data to comply with privacy regulations.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Enroll known faces (3-5 photos per person)
#    Create folders under known_faces/:
#    known_faces/kaushal/1.jpg, 2.jpg, ...
#    known_faces/dhruv/1.jpg, 2.jpg, ...

# 3. Run Phase 5 (production prototype)
python phase5_production.py                          # webcam
python phase5_production.py --source "rtsp://..."    # IP camera
python phase5_production.py --headless               # no GUI (server mode)
```

---

## Project Structure

```
├── phase1_2_capture.py       # Phase 1+2: Face detection + timestamped capture
├── phase3_recognition.py     # Phase 3:   Known vs Unknown face matching
├── phase4_integration.py     # Phase 4:   Pose + Face ID merged
├── phase5_production.py      # Phase 5:   RTSP + alerting + auto-reconnect
├── fall_detection_webcam.py  # Core fall detection state machine
├── face_recognition_webcam.py# Standalone face recognition demo
├── combined_prototype.py     # Early combined prototype
├── requirements.txt          # Python dependencies
├── known_faces/              # Enrollment photos (one folder per person)
├── captured_faces/           # Auto-saved unknown face snapshots
├── event_logs/               # Fall/intrusion event logs + snapshots
└── logs/                     # Application logs
```

---

## How It Works (Tiered Architecture)

| Tier | What it does | Runs on | Frequency |
|------|-------------|---------|-----------|
| 1 | Pose estimation (presence + posture) | CPU | Every frame |
| 2 | Face recognition (known vs unknown) | CPU | Every 15th frame |
| 3 | Fall confirmation + alert dispatch | CPU/GPU | On event only |

---

## Testing Each Phase

```bash
# Phase 1+2: Face detection + capture
python phase1_2_capture.py

# Phase 3: Face recognition
python phase3_recognition.py

# Phase 4: Combined (pose + face ID)
python phase4_integration.py

# Phase 5: Full production prototype
python phase5_production.py
```

---

## Alerting Setup (Optional)

To enable SMS/WhatsApp alerts via Twilio:

```bash
pip install twilio

# Set environment variables:
set TWILIO_ACCOUNT_SID=your_sid
set TWILIO_AUTH_TOKEN=your_token
set TWILIO_FROM_NUMBER=+1234567890

# Run with alerting:
python phase5_production.py --alert-phone "+919876543210"
```

---

## Team

- Kaushalendra Pratap Singh (AU2644013)
- Dhruv Jain (AU2644029)
- Nandish Bhatt (AU2644012)
- Dev Upadhyay (AU2644003)
