"""Compare legacy and Studio fall heuristics on a labeled manifest at UI cadence."""

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path

import cv2
import mediapipe as mp

from sentrycare.fall_detection import FallConfig, FallStateMachine, is_down_posture, full_body_features, CONFIRMED_FALL, studio_fall_config
from sentrycare.recordings import recording_frames


def main():
    cv2.setNumThreads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?", default="data/evaluation/urfd_samples.csv")
    parser.add_argument("--output", default="data/evaluation/studio_fall_comparison.json")
    args = parser.parse_args()
    manifest = Path(args.manifest).resolve()
    with manifest.open(newline="",encoding="utf-8-sig") as stream:
        clips = list(csv.DictReader(stream))
    results = []
    for clip in clips:
        path = manifest.parent / clip["path"]
        timing = manifest.parent / clip["timestamps"] if clip.get("timestamps") else None
        baseline = FallStateMachine()
        rich = FallStateMachine(studio_fall_config())
        alerts = {"baseline":None,"studio":None}
        last_sample = -1
        frames = 0
        usable = 0
        trace = []
        with mp.solutions.pose.Pose(model_complexity=1,smooth_landmarks=True,
                min_detection_confidence=.55,min_tracking_confidence=.55) as pose:
            for frame,timestamp in recording_frames(path,timing):
                if timestamp-last_sample < .2:
                    continue
                last_sample = timestamp
                result = pose.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                points = result.pose_landmarks.landmark if result.pose_landmarks else None
                old_state = baseline.update(is_down_posture(points,*frame.shape[:2]),timestamp)
                new_state = rich.update_rich(points,*frame.shape[:2],timestamp)
                if old_state==CONFIRMED_FALL and alerts["baseline"] is None:
                    alerts["baseline"] = timestamp
                if new_state==CONFIRMED_FALL and alerts["studio"] is None:
                    alerts["studio"] = timestamp
                frames+=1
                usable+=int(rich.last_pose_usable)
                trace.append({"time":timestamp,"score":rich.last_score,"state":new_state,
                              "features":full_body_features(points,*frame.shape[:2],rich.config)})
        results.append({"clip":path.name,"label":clip["label"],"frames_analyzed":frames,
                        "usable_pose_frames":usable,"first_alert_seconds":alerts,"trace":trace})
        print(json.dumps({k:v for k,v in results[-1].items() if k!="trace"}),flush=True)
    summary = {}
    for method in ("baseline","studio"):
        counts={"tp":0,"fp":0,"tn":0,"fn":0}
        for result in results:
            actual = result["label"]=="fall"
            predicted = result["first_alert_seconds"][method] is not None
            counts[("tp" if actual else "fp") if predicted else ("fn" if actual else "tn")]+=1
        summary[method]=counts
    report={"protocol":"four-clip pipeline smoke test; 5 Hz inference; no accuracy claim",
            "clips":results,"summary":summary}
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary),flush=True)


if __name__=="__main__":
    main()
