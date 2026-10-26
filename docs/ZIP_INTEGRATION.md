# ZIP integration and verification — 2026-10-27

The supplied ZIP contained the teammate's variant, its Python environment, Git metadata, and local images. Source was reviewed under ignored `tmp/friend_review/`; the archived environment, Git directory, and caches were not installed or executed. The archive was removed after integration. Personal images and embeddings remain only in ignored local storage.

## Integrated work and corrections

- Adapted normalized-embedding matching, multiple-face support, IoU association, and rich fall signals: head/hip position, descent velocity, torso orientation, temporal evidence.
- Removed the unconditional multi-detector cascade from the legacy helper. Studio uses one native YuNet pass and aligned SFace crops for enrollment and recognition, following the [OpenCV API](https://docs.opencv.org/4.x/d0/dd4/tutorial_dnn_face.html) and [official examples](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface).
- Fresh unknown results cannot indefinitely inherit an old name. Studio requires two consistent positive checks and invalidates cached identities when its gallery changes.
- Studio's face recognition is independent of pose presence, so close-up faces can be recognized without visible hips.
- Single-face, sharpness, exposure, and minimum-size validation provides explicit import/upload errors.
- A local SQLite registry stores aligned images and versioned numeric embeddings. DeepFace pickle caches are not consulted during Studio processing.
- Preview is independent of inference. There is one analysis request in flight, a separate embedding cadence, and no frame backlog.
- Previous default fall thresholds were retained for compatibility. Studio uses an explicit 1.2-second window and three-frame motion history for its approximately 5 Hz analysis cadence. The unreachable-score bug for sustained horizontal torso evidence was fixed. This is not evidence of validated fall accuracy.

## Actual checks

- Of 15 archived enrollment images, twelve passed quality checks across five local profiles. Three profiles have three usable samples; two need more images. Rejected images had a small face or multiple faces. Names, photos, feature vectors, and person-level diagnostics are excluded from the PR.
- All twelve accepted images were indexed. Only people with at least three built samples are active.
- A leave-one-sample-out check on the three complete profiles matched 9/9 samples using their remaining images as references. With each tested identity withheld entirely, 9/9 were rejected as unknown. These are small within-enrollment consistency checks, not a real-world accuracy benchmark. The local diagnostic is `tmp/studio_consistency.json`.
- All four tabs passed browser layout/navigation checks at 1440px and 390px, without horizontal overflow or JavaScript exceptions.
- A temporary registry and prerecorded virtual camera exercised real browser capture at three angle labels, YuNet/SFace feature building, named recognition, gallery images, deletion-dialog cancellation, and camera shutdown. One recognized analysis frame reported 125 ms. This is a fixture measurement, not a physical live-person or target-hardware benchmark.
- That browser fixture processed MediaPipe posture simultaneously. Face recognition succeeded without a usable full-body pose.
- The 69 feature regression tests passed. Three additional repository privacy tests guard ignored files and the tracked file list. The package build was checked to include its HTML, JavaScript, and CSS assets, and installed dependencies passed compatibility checks.
- At 5 Hz on the four existing URFD clips, the corrected Studio heuristic detected `fall-02` at 3.637 seconds from clip start and kept both ADLs negative. `fall-01` remained missed: only one reliable down-pose sample was available before landmark loss. The legacy configuration missed both falls. This is a development smoke test used while adjusting the implementation, not an independent accuracy benchmark. The full diagnostic is `data/evaluation/studio_fall_comparison.json`.

Fixture state and screenshots are ignored under `tmp/`; no test profile was added to the main registry. Automated test totals are recorded in the task handoff.

## Remaining validation

Identity matching uses a cosine-similarity gate of 0.45 and competing-identity margin of 0.06. These are separate from the teammate's DeepFace cosine-distance thresholds because the preprocessing pipelines differ. Broader identity/unknown benchmarks and new live images under different lighting are still required.

Fall monitoring follows one body. People overlapping or crossing can invalidate temporal motion evidence; multi-person body tracking and benchmarked classification remain future work. Event names are assigned only when a recognized face overlaps the visible pose head.

In particular, the remaining occluded fall demonstrates that a geometry-only pipeline is not sufficient for deployment. A trained temporal model, additional person tracking, and more independent recordings are required before making performance claims. A pose-loss warning does not count as a confirmed fall.
