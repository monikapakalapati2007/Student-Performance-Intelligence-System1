"""Local web application for the Student Performance Intelligence System.

Standard library only — no Flask/FastAPI install required.

    .venv\\Scripts\\python.exe app\\server.py            # http://127.0.0.1:8000
    .venv\\Scripts\\python.exe app\\server.py --port 8080 --open

The server renders ``app/static`` and exposes:

    GET  /api/health     service + Ollama + model status
    GET  /api/meta       measured metrics, feature importance, dataset stats
    POST /api/predict    {study_hours, attendance, participation} -> grounded prediction
    POST /api/ask        {question, context} -> Ollama explanation over model facts
    POST /api/exams      {gender, race/ethnicity, ...} -> secondary exams model

Every number shown in the UI is produced by the trained joblib models or read from
``results/``; the language model only writes prose around those numbers.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import threading
import traceback
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Tuple

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
STATIC_DIR = os.path.join(APP_DIR, "static")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from insight import InsightEngine  # noqa: E402
from ollama_service import OllamaService  # noqa: E402

MAX_BODY_BYTES = 256_000


class Application:
    """Loads the models once and shares them across request threads."""

    def __init__(self) -> None:
        self.engine = InsightEngine()
        self.ollama = OllamaService(self.engine)
        self.lock = threading.Lock()
        self.metrics = self.engine.metrics

    # -- endpoints ---------------------------------------------------------
    def health(self) -> Dict[str, Any]:
        status = self.ollama.status(refresh=True)
        return {
            "status": "ok",
            "model": self.engine.model_config.get("name", "hist_gradient_boosting"),
            "artifacts": bool(self.engine.model_config),
            "ollama": status,
            "mode": "ollama" if (status["available"] and status["model"]) else "deterministic",
        }

    def meta(self) -> Dict[str, Any]:
        return {
            "project": "Student Performance Intelligence System",
            "primary_dataset": self.metrics.get("primary_dataset", "Student Performance 1M"),
            "task": self.metrics.get("task", ""),
            "metrics": self.engine.model_metrics(),
            "importance": self.engine.feature_importance(),
            "dataset": self.engine.dataset_statistics(),
            "limitations": self.engine.limitations(),
            "grade_cutoffs": {"A": 85, "B": 70, "C": 55, "D": 40},
            "features": ["weekly_self_study_hours", "attendance_percentage",
                         "class_participation"],
            "candidate_comparison": self.metrics.get("candidates", {}),
            "ollama": self.ollama.status(),
        }

    def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        hours = payload.get("study_hours", payload.get("weekly_self_study_hours"))
        attendance = payload.get("attendance", payload.get("attendance_percentage"))
        participation = payload.get("participation", payload.get("class_participation"))
        if hours is None or attendance is None or participation is None:
            raise ValueError("study_hours, attendance and participation are all required")
        with self.lock:
            return self.engine.predict(hours, attendance, participation)

    def ask(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        question = (payload.get("question") or "").strip()
        if not question:
            raise ValueError("question is required")
        context = payload.get("context") or None
        if context is not None and not isinstance(context, dict):
            raise ValueError("context must be an object")
        with self.lock:
            return self.ollama.answer(question, context,
                                      use_llm=bool(payload.get("use_llm", True)),
                                      use_tools=bool(payload.get("use_tools", False)))

    def exams(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self.lock:
            return self.engine.predictor.predict_exams(payload)


class Handler(BaseHTTPRequestHandler):
    server_version = "StudentPerformanceApp/1.0"
    application: Application

    # -- utilities ---------------------------------------------------------
    def _send(self, status: int, body: bytes, content_type: str,
              extra: Dict[str, str] = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, status: int, obj: Any) -> None:
        # allow_nan=False: a NaN would make the response invalid JSON for the browser.
        body = json.dumps(obj, ensure_ascii=False, default=str,
                          allow_nan=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _error(self, status: int, message: str) -> None:
        self._json(status, {"error": message, "status": status})

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        if length > MAX_BODY_BYTES:
            raise ValueError("request body too large")
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ValueError("body must be valid JSON") from None
        if not isinstance(data, dict):
            raise ValueError("body must be a JSON object")
        return data

    # -- routes ------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        route = urllib.parse.urlparse(self.path).path
        try:
            if route in ("/", "/index.html"):
                return self._static("index.html")
            if route == "/api/health":
                return self._json(200, self.application.health())
            if route == "/api/meta":
                return self._json(200, self.application.meta())
            if route.startswith("/api/"):
                return self._error(404, f"unknown endpoint {route}")
            return self._static(route.lstrip("/"))
        except BrokenPipeError:
            return
        except Exception:  # pragma: no cover - defensive
            traceback.print_exc()
            return self._error(500, "internal server error")

    def do_HEAD(self) -> None:  # noqa: N802
        self.do_GET()

    def do_POST(self) -> None:  # noqa: N802
        route = urllib.parse.urlparse(self.path).path
        handlers = {"/api/predict": self.application.predict,
                    "/api/ask": self.application.ask,
                    "/api/exams": self.application.exams}
        if route not in handlers:
            return self._error(404, f"unknown endpoint {route}")
        try:
            payload = self._read_json()
            return self._json(200, handlers[route](payload))
        except ValueError as exc:
            return self._error(400, str(exc))
        except BrokenPipeError:
            return
        except Exception as exc:  # pragma: no cover - defensive
            traceback.print_exc()
            return self._error(500, f"{type(exc).__name__}: {exc}")

    def _static(self, relative: str) -> None:
        relative = relative.replace("\\", "/")
        if ".." in relative.split("/") or relative.startswith("/"):
            return self._error(403, "forbidden path")
        path = os.path.join(STATIC_DIR, relative or "index.html")
        if not os.path.isfile(path):
            return self._error(404, f"not found: {relative}")
        content_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type in (
                "application/javascript", "application/json"):
            content_type += "; charset=utf-8"
        with open(path, "rb") as fh:
            body = fh.read()
        self._send(200, body, content_type)

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


def build_server(port: int, host: str = "127.0.0.1") -> Tuple[ThreadingHTTPServer, Application]:
    application = Application()
    handler = type("BoundHandler", (Handler,), {"application": application})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server, application


def main() -> int:
    parser = argparse.ArgumentParser(description="Student Performance web application.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--open", action="store_true", help="open the browser on startup")
    args = parser.parse_args()

    server, application = build_server(args.port, args.host)
    url = f"http://{args.host}:{args.port}/"
    status = application.ollama.status(refresh=True)
    print(f"Student Performance Intelligence System -> {url}")
    print(f"  primary model : {application.engine.model_config.get('name')}")
    print(f"  Ollama        : {'available, model ' + str(status['model']) if status['available'] and status['model'] else 'not available -> deterministic mode'}")
    print("  Ctrl+C to stop")
    if args.open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
