# SentryCare Studio

## Open one application

Double-click `Start SentryCare.cmd` in the project folder, or run:

```powershell
.\.venv\Scripts\python.exe -m sentrycare.web
```

The double-click launcher opens your browser automatically. You can also open **http://127.0.0.1:8765** in Chrome or Edge. Keep the terminal open; Ctrl+C stops the server. The app listens only on this computer. Browser camera access is requested when you click **Start camera**.

All daily tasks are in the UI. `scripts/run.ps1` also opens Studio by default. The earlier webcam script remains available with `scripts/run.ps1 -Mode webcam`.

## Enroll someone

1. Open **Enroll a person**. Enter their name and click **Create profile**, or select an existing person.
2. Click **Start camera** and allow camera access. Use good lighting, one person in view, and a slight turn rather than a full profile.
3. Select **Front**, **Left**, **Right**, **Slightly up**, or **Slightly down**, and click **Capture photo**. The server independently validates every saved image. Photos with multiple faces, a tiny face, excessive blur, or poor lighting are rejected with an explanation.
4. Save at least three clear photos; five varied angles are preferable. You can upload existing photos in the same tab.
5. Click **Build profile**. This calculates aligned SFace embeddings and saves them locally; it does not retrain SFace's neural-network weights. A ready profile becomes available immediately.

Adding photos to a ready profile marks it **Update needed**. Build again to include the new images. Only profiles with at least three built samples are eligible for recognition.

## Review saved data

**Saved profiles** shows each name, photo count, training status, and image previews. **View & add photos** shows all images and their capture-angle labels. You can remove individual photos or a profile after confirmation. Removal also removes its face features. Falling below three built images removes the profile from the active matcher.

**Import existing photos** reads eligible images from `data/known_faces/<name>/` and, if present, `facial_reco_cctv.zip`. It never executes archived scripts, reads archived pickle caches, or installs the archived environment. Imports are deduplicated. Original files are preserved. Review rejection reasons, then click **Build all profiles**.

### Local import results

Fifteen enrollment images were checked; twelve passed quality checks and were indexed locally. Three profiles are ready. Two profiles need additional images: one needs at least one clear photo, and the other needs at least two single-person photos. Inspect the local UI for names and images; personal enrollment details are not included in this repository.

One image had a face too small for enrollment. Two contained multiple detected faces. Group photos cannot reliably label an individual profile. The supplied ZIP was removed after integration; the accepted images and embeddings remain only in ignored local storage.

## Recognize people and monitor posture

Open **Live monitoring** and start the camera. Face recognition works independently of body detection, including close-ups. A name needs two consistent embedding checks. Weak or competing matches return **Unknown**; unusable images show **Checking** with an explanation.

Similarity is a raw cosine score, not a probability or percentage accuracy.

Keep shoulders and hips visible for body monitoring. Studio combines torso orientation, head/hip position, recent descent, and temporal confirmation. It follows one body, even when multiple faces are detected. Events get a name only when a recognized face can be associated with that body's head; otherwise they are unassigned.

The browser shows a direct preview. Frames are analyzed at a bounded cadence with one request in flight; slow analysis does not queue old frames or pause the visible preview. Embeddings refresh less frequently than face positions. Studio performs no cloud inference or notification dispatch.

**Event history** lists locally recorded posture events. Fall detection remains a research heuristic and needs full dataset and live-environment validation. Do not stage dangerous falls.

The current four-clip check detected one of two falls, with no alerts on the two ADL clips. The remaining fall lost usable body landmarks during the confirmation window. This is a known limitation, not a validated reliability estimate.

## Storage and troubleshooting

- `data/studio/enrollment.sqlite3`: profile metadata, versioned embeddings, events.
- `data/studio/images/`: aligned face images shown in the gallery.
- `data/models/`: official YuNet and SFace weights; initial startup can download missing assets.
- ZIP archives, if supplied for optional imports, are ignored by Git. The original integration archive was removed.

Photos and embeddings remain local and are excluded from Git. Enroll people with their permission. If the camera is busy, close video-call apps or old OpenCV scripts. Use the camera selector for another device.

If models cannot load, the UI shows the initialization error. Check internet access for initial downloads and restart after correcting the issue. Refresh the page after restarting the server to obtain its new local request token.

For a fresh checkout, run `scripts/setup.ps1` first. Local environments and assets are not committed. See `TEAMMATE_SETUP.md` for installation; use Studio for enrollment and daily monitoring.
