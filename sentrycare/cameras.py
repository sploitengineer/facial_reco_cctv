"""Headless camera workers with bounded restarts, progress watchdog, and status JSON."""

import argparse
from dataclasses import dataclass
import json
import logging
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from .paths import PROJECT_ROOT, LOGS, ensure_data_dirs

logger = logging.getLogger("SentryCare")


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config.get("cameras"), list) or not config["cameras"]:
        raise ValueError("Configuration needs a nonempty cameras list")
    ids, sources = set(), set()
    enabled = []
    for camera in config["cameras"]:
        name = camera.get("id", "")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name) or name in ids:
            raise ValueError("Camera IDs must be unique safe directory names")
        ids.add(name)
        if not isinstance(camera.get("enabled", True), bool) or not isinstance(camera.get("face_id", True), bool):
            raise ValueError("enabled and face_id must be JSON booleans")
        if not camera.get("enabled", True):
            continue
        if ("source" in camera) == ("source_env" in camera):
            raise ValueError(f"Camera {name} requires exactly one of source or source_env")
        source = os.environ.get(camera["source_env"]) if "source_env" in camera else camera["source"]
        if isinstance(source, bool) or not isinstance(source, (int, str)) or source == "" or (isinstance(source, int) and source < 0):
            raise ValueError(f"Camera {name} needs a source; set its configured environment variable")
        source = str(source)
        if source in sources:
            raise ValueError("Enabled cameras must use different sources")
        sources.add(source)
        enabled.append({**camera, "source": source})
    if not enabled:
        raise ValueError("Enable at least one camera")
    limit = config.get("restart_limit", 3)
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError("restart_limit must be a nonnegative integer")
    for key, default in (("restart_delay", 5), ("startup_timeout", 300), ("stall_timeout", 60)):
        value = config.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
        config[key] = value
    return {**config, "restart_limit": limit, "cameras": enabled}


def worker_command(camera, heartbeat, fall_config=None):
    command = [sys.executable, "-m", "sentrycare.app", "--source", camera["source"],
               "--headless", "--camera-id", camera["id"], "--heartbeat", str(heartbeat)]
    if not camera.get("face_id", True):
        command.append("--no-face-id")
    if fall_config:
        command.extend(["--fall-config", str(fall_config)])
    return command


@dataclass
class Worker:
    camera: dict
    heartbeat: Path
    process: object = None
    log_handle: object = None
    started_at: float = 0
    restarts: int = 0
    next_start: float = 0
    state: str = "pending"
    exit_code: int = None


def stalled(worker, now, startup_timeout, stall_timeout):
    if worker.heartbeat.exists():
        modified = worker.heartbeat.stat().st_mtime
        if modified >= worker.started_at:
            return now - modified > stall_timeout
    return now - worker.started_at > startup_timeout


def stop_worker(worker):
    if worker.process is not None and worker.process.poll() is None:
        worker.process.terminate()
        try:
            worker.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            worker.process.kill()
            worker.process.wait(timeout=10)
    if worker.log_handle is not None:
        worker.log_handle.close()
        worker.log_handle = None


def supervise(config, fall_config=None, duration=None):
    ensure_data_dirs()
    workers = [Worker(camera, LOGS / f"heartbeat_{camera['id']}.json") for camera in config["cameras"]]
    status = LOGS / "cameras_status.json"
    deadline = time.monotonic() + duration if duration else None
    try:
        while deadline is None or time.monotonic() < deadline:
            now = time.time()
            for worker in workers:
                if worker.state == "exhausted":
                    continue
                if worker.process is None:
                    if now < worker.next_start:
                        continue
                    worker.log_handle = (LOGS / f"worker_{worker.camera['id']}.log").open("a", encoding="utf-8")
                    worker.started_at = now
                    worker.process = subprocess.Popen(
                        worker_command(worker.camera, worker.heartbeat, fall_config), cwd=PROJECT_ROOT,
                        stdout=worker.log_handle, stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                    )
                    worker.state = "starting"
                    logger.info("Started camera %s", worker.camera["id"])
                code = worker.process.poll()
                if code is None and stalled(worker, now, config["startup_timeout"], config["stall_timeout"]):
                    logger.warning("Camera %s stalled; terminating worker", worker.camera["id"])
                    stop_worker(worker)
                    code = worker.process.poll()
                if code is not None:
                    stop_worker(worker)
                    worker.exit_code = code
                    worker.process = None
                    if worker.restarts >= config["restart_limit"]:
                        worker.state = "exhausted"
                    else:
                        worker.restarts += 1
                        worker.next_start = now + config["restart_delay"]
                        worker.state = "retrying"
                    logger.warning("Camera %s exited (%s); %s", worker.camera["id"], code, worker.state)
                elif worker.heartbeat.exists() and worker.heartbeat.stat().st_mtime >= worker.started_at:
                    worker.state = "processing"
            temp = status.with_suffix(".tmp")
            temp.write_text(json.dumps({"updated_at": now, "cameras": [
                {"id": w.camera["id"], "status": w.state, "restarts": w.restarts, "exit_code": w.exit_code}
                for w in workers]}, indent=2), encoding="utf-8")
            temp.replace(status)
            if all(w.state == "exhausted" for w in workers):
                return 1
            time.sleep(0.5)
    except KeyboardInterrupt:
        logger.info("Stopping camera workers")
    finally:
        for worker in workers:
            stop_worker(worker)
        temp = status.with_suffix(".tmp")
        temp.write_text(json.dumps({"updated_at": time.time(), "cameras": [
            {"id": w.camera["id"], "status": "exhausted" if w.state == "exhausted" else "stopped",
             "restarts": w.restarts, "exit_code": w.exit_code} for w in workers]}, indent=2), encoding="utf-8")
        temp.replace(status)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/cameras.json"))
    parser.add_argument("--fall-config")
    parser.add_argument("--check", action="store_true", help="Validate configuration without opening cameras")
    parser.add_argument("--duration", type=float, help="Stop the supervisor after this many seconds")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
            raise ValueError("Duration must be finite and positive")
        if args.fall_config:
            from .fall_detection import FallConfig
            FallConfig.load(args.fall_config)
    except (ValueError, OSError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    if args.check:
        print(f"Configuration valid: {len(config['cameras'])} enabled camera(s)")
        return 0
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    return supervise(config, args.fall_config, args.duration)


if __name__ == "__main__":
    raise SystemExit(main())
