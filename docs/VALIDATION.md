# Validation results: 2026-10-05

## Installed runtime

Local Python 3.12.12, MediaPipe 0.10.21, OpenCV 4.11.0.86, DeepFace 0.0.93, TensorFlow 2.18.0, and NumPy 1.26.4. All 82 installed packages passed `uv pip check`. SentryCare is installed as an editable package.

## Inference and camera integration

- MediaPipe initialized and processed a synthetic blank frame.
- DeepFace imported, downloaded SFace into `data/models/`, and produced a 128-dimensional embedding from synthetic input. This verifies model execution, not face-match accuracy.
- Camera 0 processed 30 frames in 1.73 seconds through MediaPipe. No pose was found in those frames. The test saved no camera images and sent no notifications.
- The camera supervisor ran for 25 seconds, processed at least 167 frames after startup, wrote camera-specific heartbeat/status/log files, and stopped with zero restarts. Its final status is `stopped`.
- Windows camera access required execution outside the sandbox during this session. The ordinary local launch commands can be used from the user's terminal.

Reports are in `data/logs/runtime_check.json`, `cameras_status.json`, and `heartbeat_laptop.json`.

## Official URFD sample evaluation

Source: [University of Rzeszow UR Fall Detection Dataset](https://fenix.ur.edu.pl/~mkepski/ds/uf.html), CC BY-NC-SA 4.0 for non-commercial academic use. Attribution is retained in `data/datasets/urfd_samples/ATTRIBUTION.md`.

Only four camera-0 RGB sequences were used, with the official frame-number/millisecond CSVs. No frame rate was guessed. These are clip-number splits for pipeline verification; they are not established as subject-disjoint splits.

| Recording | Split | Frames | Frames with usable torso features |
|---|---|---:|---:|
| fall-01 | train | 160 | 133 |
| adl-01 | train | 150 | 136 |
| fall-02 | validation | 110 | 110 |
| adl-02 | validation | 180 | 139 |

The 27-setting grid covered aspect ratios 1.1/1.3/1.5, normalized vertical positions 0.45/0.55/0.65, and confirmation durations 1/2/3 seconds.

Every candidate missed the training fall; training F1 was 0. The deterministic tie selected aspect ratio 1.1, position 0.45, and duration 1 second. With that candidate, the two validation clips yielded one detected fall and one correctly negative ADL, with first alert at 2.869 seconds from clip start. There is no fall-onset annotation in this evaluation, so this is not measured alert latency.

The perfect score on just two held-out clips does not offset the training miss or establish generalization. The tool flags zero training detections. The exported candidate was not applied to the live camera, and default thresholds remain unchanged. The next detection task is analysis of the missed fall and a broader, properly separated benchmark.

Generated artifacts: `data/evaluation/urfd_samples.csv`, `urfd_features.json`, `urfd_report.json`, and `urfd_candidate.json`.

## Tests and practical limits

All 29 regression tests passed. They check fall timing and recovery, missing/low-confidence poses, evaluation split separation, CSV logging, RTSP reconnects, worker isolation, bounded restarts, watchdog timestamps, shutdown, and sequence frame ordering. Syntax checks passed for 37 Python files and both PowerShell scripts; packaged and compatibility CLI entry points passed their help checks.

Unverified: real enrolled-face matching, multiple physical cameras at once, live RTSP connectivity, SMS delivery, full-dataset accuracy, multi-person tracking, GPU event confirmation, and cloud deployment.
