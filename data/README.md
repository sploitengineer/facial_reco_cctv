Runtime files stay here and are excluded from version control.

- `known_faces/<name>/`: 3-5 enrollment photos per consenting person.
- `captured_faces/<camera-id>/`: unknown-face snapshots.
- `event_logs/<camera-id>/`: confirmed-posture snapshots and CSV event logs.
- `logs/`: application logs, runtime checks, supervisor status, and worker heartbeats.
- `datasets/`: local public research datasets and their attribution.
- `evaluation/`: manifests, cached features, threshold configurations, reports.

Existing captured images were moved into `captured_faces/` without modification.
Override this root with `SENTRYCARE_DATA_DIR` before launching a process.
