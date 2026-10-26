# What to do next

1. Start Studio with `Start SentryCare.cmd` or `.venv/Scripts/python -m sentrycare.web`, then open http://127.0.0.1:8765. Allow the camera when you click **Start camera**.
2. Open **Saved profiles**. Imported images and features stay on the original machine; they are not included in a Git checkout. On that machine, three profiles are ready and two need additional photos. A teammate's fresh checkout starts without personal enrollment data.
3. In **Enroll a person**, select the incomplete person, capture front/left/right/slight-up/slight-down photos, and click **Build profile**. The UI explains any rejected samples. For a new person, enter their name first.
4. Open **Live monitoring**. Check known names, an unenrolled person's **Unknown** result, and normal movements under your actual lighting. Face recognition works without a full-body pose. For fall monitoring, keep shoulders and hips visible. Stop the camera using either camera control, including the global control visible on every tab.
5. Use **Saved profiles** to inspect/remove photos and **Event history** to review posture events. All biometric data stays local under `data/studio/` and is excluded from Git.

Next engineering work: measure live face accuracy under varied lighting, expand the identity/unknown benchmark, investigate fall misses across the full datasets, and add multi-person body tracking. The present fall heuristic is not deployment-validated; do not stage dangerous falls for testing.

See [STUDIO_GUIDE.md](STUDIO_GUIDE.md) for detailed steps and [ZIP_INTEGRATION.md](ZIP_INTEGRATION.md) for changes and verification. RTSP cameras and external notifications remain separate optional integrations; Studio does not dispatch SMS.
