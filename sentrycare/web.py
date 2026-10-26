"""Local SentryCare Studio: camera enrollment, profile gallery, and live monitoring."""

import argparse
from pathlib import Path
import re
import secrets
import threading
import webbrowser

from flask import Flask, jsonify, request, send_file, render_template
from werkzeug.exceptions import HTTPException

from .paths import DATA_DIR, PROJECT_ROOT
from .studio import Studio, decode_image


def create_app(studio=None):
    app = Flask(__name__, template_folder="web/templates", static_folder="web/static", static_url_path="/static")
    app.config.update(MAX_CONTENT_LENGTH=8 * 1024 * 1024, JSON_SORT_KEYS=False)
    app.json.sort_keys = False
    service = studio or Studio(DATA_DIR / "studio")
    app.extensions["studio"] = service
    token = secrets.token_urlsafe(32)

    @app.before_request
    def local_requests():
        host = request.host.split(":")[0]
        if host not in ("127.0.0.1", "localhost", "[::1]", "::1"):
            return jsonify(error="Open Studio through localhost"), 403
        if request.method in ("POST", "PUT", "PATCH", "DELETE") and not secrets.compare_digest(
                request.headers.get("X-SentryCare-Token", ""), token):
            return jsonify(error="Refresh the page before making changes"), 403

    @app.after_request
    def private_responses(response):
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = ("default-src 'self'; script-src 'self'; "
            "style-src 'self' 'unsafe-inline'; img-src 'self' blob: data:; media-src 'self' blob:; "
            "connect-src 'self'; frame-ancestors 'none'")
        return response

    @app.errorhandler(Exception)
    def errors(exc):
        if isinstance(exc, HTTPException):
            return jsonify(error=exc.description), exc.code
        if isinstance(exc, ValueError):
            return jsonify(error=str(exc)), 400
        if isinstance(exc, RuntimeError):
            return jsonify(error=str(exc)), 503
        app.logger.exception("Studio request failed")
        return jsonify(error="The operation failed. Check the local server log and try again."), 500

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/state")
    def state():
        return jsonify(token=token, engine={"status": service.status, "error": service.error},
                       people=service.registry.people(), settings=service.settings,
                       job=service.jobs.get(service.active_job), model="SFace · YuNet aligned faces",
                       import_available=(PROJECT_ROOT / "facial_reco_cctv.zip").is_file()
                           or any((PROJECT_ROOT / "data/known_faces").glob("*/*")))

    @app.post("/api/people")
    def create_person():
        return jsonify(service.registry.create_person((request.get_json(silent=True) or {}).get("name", ""))), 201

    @app.delete("/api/people/<person_id>")
    def delete_person(person_id):
        if not service.registry.delete_person(person_id):
            return jsonify(error="Profile not found"), 404
        return jsonify(ok=True)

    @app.post("/api/people/<person_id>/samples")
    def add_sample(person_id):
        image = request.files.get("image")
        data = image.read() if image else request.get_data()
        angle = request.form.get("angle", request.args.get("angle", "front"))
        sample = service.enroll(person_id, decode_image(data), angle)
        return jsonify(sample_id=sample, person=service.registry.person(person_id)), 201

    @app.delete("/api/samples/<sample_id>")
    def delete_sample(sample_id):
        if not service.registry.delete_sample(sample_id):
            return jsonify(error="Photo not found"), 404
        return jsonify(ok=True)

    @app.get("/api/samples/<sample_id>/image")
    def sample_image(sample_id):
        if not re.fullmatch(r"[0-9a-f]{32}", sample_id):
            return jsonify(error="Photo not found"), 404
        path = service.registry.images / f"{sample_id}.jpg"
        if not path.is_file():
            return jsonify(error="Photo not found"), 404
        return send_file(path, mimetype="image/jpeg")

    @app.post("/api/train")
    def train():
        person_id = (request.get_json(silent=True) or {}).get("person_id")
        return jsonify(service.start_job("train", person_id)), 202

    @app.post("/api/import")
    def import_existing():
        return jsonify(service.start_job("import")), 202

    @app.get("/api/jobs/<job_id>")
    def job(job_id):
        if job_id not in service.jobs:
            return jsonify(error="Job not found"), 404
        return jsonify(service.jobs[job_id])

    @app.post("/api/frame")
    def frame():
        session_id = request.headers.get("X-Session-ID", "")
        if not re.fullmatch(r"[a-zA-Z0-9_-]{8,64}", session_id):
            raise ValueError("Invalid camera session; refresh the page")
        mode = request.args.get("mode", "monitor")
        if mode not in ("monitor", "enroll"):
            raise ValueError("Invalid camera mode")
        result = service.analyze(decode_image(request.get_data()), session_id, mode)
        return (jsonify(busy=True), 202) if result is None else jsonify(result)

    @app.get("/api/events")
    def events():
        return jsonify(events=service.registry.events())

    return app


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open-browser", action="store_true", help="Open the local UI after starting")
    args = parser.parse_args(argv)
    if not 1024 <= args.port <= 65535:
        parser.error("Choose a port between 1024 and 65535")
    print(f"SentryCare Studio: http://127.0.0.1:{args.port}", flush=True)
    app = create_app()
    if args.open_browser:
        opener = threading.Timer(1, lambda: webbrowser.open(f"http://127.0.0.1:{args.port}"))
        opener.daemon = True
        opener.start()
    app.run(host="127.0.0.1", port=args.port, threaded=True, use_reloader=False, debug=False)


if __name__ == "__main__":
    main()
