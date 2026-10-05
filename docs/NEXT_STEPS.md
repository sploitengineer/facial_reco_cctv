# Steps requiring your input

The Python environment, package installation, camera-0 configuration, model download, and official four-clip URFD sample download have been handled locally. No browser account is needed for those steps.

1. **Try the live view:** run `scripts/run.ps1` from the project root. Keep shoulders and hips in view; the webcam tests processed frames but found no body pose. Press `q` to stop. Test with normal movements rather than staging a dangerous fall.
2. **Enroll known people:** place 3-5 consented reference photos per person in `data/known_faces/<name>/`, then run `scripts/run.ps1 -FaceId`. Real identity accuracy has not been measured because no reference photos are enrolled.
3. **Add an IP camera when ready:** supply its RTSP address using the environment variable in `configs/cameras.json`, and enable that camera. A laptop webcam is the only live source tested so far.
4. **Optional real notifications:** provide your own Twilio sender/account and destination using shell variables. No credentials, account creation, or external notification dispatch was performed.
5. **Full research evaluation:** obtain the complete datasets under their usage terms and arrange subject/session-separated manifests. The URFD samples are for pipeline verification. The old official Le2i URL was inaccessible during this session; its availability needs checking before making a full-dataset plan.

The default fall thresholds remain unvalidated for deployment. A full dataset benchmark and real-world false-positive review should precede changing deployment settings.
