"""Register the project's Ollama model and check the local Ollama installation.

    python src/ollama_setup.py --status             # is Ollama installed/running? which models?
    python src/ollama_setup.py --pull               # download the small base model (~1.3 GB)
    python src/ollama_setup.py --write-modelfile    # render models/ollama/Modelfile.student-performance
    python src/ollama_setup.py --create             # register student-performance-advisor
    python src/ollama_setup.py --all                # pull + modelfile + create

The Modelfile is generated from ``src/ollama_prompt.py`` so the file on disk and the
registered model always carry the same system prompt and parameters.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from ollama_prompt import (BASE_MODEL, MODELFILE_PATH, MODEL_NAME, PARAMETERS,  # noqa: E402
                           TEMPLATE, SYSTEM_PROMPT, render_modelfile)
from ollama_service import OllamaService, OllamaUnavailable  # noqa: E402


def find_ollama_executable():
    candidates = [shutil.which("ollama"),
                  os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe"),
                  os.path.expandvars(r"%PROGRAMFILES%\Ollama\ollama.exe"),
                  "/usr/local/bin/ollama", "/usr/bin/ollama"]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def cmd_status(service: OllamaService) -> int:
    exe = find_ollama_executable()
    status = service.status(refresh=True)
    print(json.dumps({"executable": exe, **status}, indent=2))
    if not status["available"]:
        print("\nOllama is not reachable. On Windows start the desktop app (or run "
              "`ollama serve`), then retry.", file=sys.stderr)
        return 1
    return 0


def cmd_pull(service: OllamaService) -> int:
    print(f"Pulling {BASE_MODEL} (this can take a few minutes on a slow connection)...")
    try:
        service._request("/api/pull", {"model": BASE_MODEL, "stream": False}, timeout=1800)
    except OllamaUnavailable as exc:
        print(f"pull failed: {exc}", file=sys.stderr)
        return 1
    print(f"{BASE_MODEL} is available.")
    return 0


def cmd_write_modelfile() -> int:
    os.makedirs(os.path.dirname(MODELFILE_PATH), exist_ok=True)
    with open(MODELFILE_PATH, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(render_modelfile())
    print(f"wrote {os.path.relpath(MODELFILE_PATH, PROJECT_ROOT)}")
    return 0


def cmd_create(service: OllamaService) -> int:
    payload = {
        "model": MODEL_NAME,
        "from": BASE_MODEL,
        "system": SYSTEM_PROMPT,
        "template": TEMPLATE,
        "parameters": PARAMETERS,
        "stream": False,
    }
    try:
        service._request("/api/create", payload, timeout=600)
    except OllamaUnavailable as exc:
        print(f"create failed: {exc}", file=sys.stderr)
        return 1
    print(f"registered Ollama model '{MODEL_NAME}' from {BASE_MODEL}")
    status = service.status(refresh=True)
    print("models:", ", ".join(status["models"]) or "(none)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Set up the Ollama model for this project.")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--pull", action="store_true")
    parser.add_argument("--write-modelfile", action="store_true")
    parser.add_argument("--create", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    service = OllamaService()
    if args.write_modelfile or args.all:
        if cmd_write_modelfile():
            return 1
    if args.pull or args.all:
        if cmd_pull(service):
            return 1
    if args.create or args.all:
        if cmd_create(service):
            return 1
    if args.status or args.all or not any(vars(args).values()):
        return cmd_status(service)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
