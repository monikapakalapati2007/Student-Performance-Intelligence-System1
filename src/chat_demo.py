"""Dependency-free chat demo for the Air Quality SLM.

    python src/chat_demo.py            # then open http://127.0.0.1:8080

Serves a browser chat UI and an **Ollama-compatible REST API**
(`/api/chat`, `/api/generate`, `/api/tags`, `/api/ps`), so the same frontend and
any Ollama client work whether the model is served from this process or from
`ollama serve`. Uses only the Python standard library for the server.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import torch  # noqa: E402

from inference import STANDARD_QUESTIONS, generate_answer, load_questions  # noqa: E402
from model import load_model_for_inference  # noqa: E402
from profiles import default_models_dir, read_model_profile  # noqa: E402

STATE: Dict[str, Any] = {"model": None, "tokenizer": None, "config": None,
                         "meta": None, "device": None, "name": "air-quality-slm",
                         "lock": threading.Lock()}

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Air Quality SLM</title>
<style>
  :root { --bg:#0f1720; --panel:#17222e; --ink:#e8eef5; --muted:#8fa3b8;
          --accent:#3fa9f5; --user:#1f3b52; --bot:#1b2836; }
  * { box-sizing:border-box; }
  body { margin:0; font:15px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;
         background:var(--bg); color:var(--ink); height:100vh; display:flex; flex-direction:column; }
  header { padding:14px 18px; border-bottom:1px solid #24323f; display:flex;
           align-items:baseline; gap:10px; }
  header h1 { font-size:16px; margin:0; font-weight:600; }
  header span { font-size:12px; color:var(--muted); }
  #log { flex:1; overflow-y:auto; padding:18px; display:flex; flex-direction:column; gap:12px; }
  .msg { max-width:80%; padding:10px 13px; border-radius:12px; white-space:pre-wrap; }
  .user { align-self:flex-end; background:var(--user); }
  .bot { align-self:flex-start; background:var(--bot); border:1px solid #24323f; }
  .role { font-size:11px; text-transform:uppercase; letter-spacing:.06em;
          color:var(--muted); margin-bottom:3px; }
  footer { padding:12px 18px; border-top:1px solid #24323f; display:flex; gap:10px; }
  textarea { flex:1; resize:none; height:46px; padding:12px; border-radius:10px;
             border:1px solid #2a3a4a; background:#101a24; color:var(--ink); font:inherit; }
  button { padding:0 20px; border-radius:10px; border:0; background:var(--accent);
           color:#04121c; font-weight:700; cursor:pointer; }
  button:disabled { opacity:.5; cursor:default; }
  .chips { padding:0 18px 10px; display:flex; flex-wrap:wrap; gap:8px; }
  .chip { font-size:12px; padding:6px 10px; border-radius:999px; background:#16232f;
          border:1px solid #24323f; color:var(--muted); cursor:pointer; }
  .chip:hover { color:var(--ink); border-color:var(--accent); }
</style>
</head>
<body>
<header><h1>Air Quality SLM</h1><span id="status">loading model…</span></header>
<div id="log">
  <div class="msg bot"><div class="role">assistant</div>Ask me about the air quality
  measurements in the dataset — pollutant levels, weather context, or air quality categories.</div>
</div>
<div class="chips" id="chips"></div>
<footer>
  <textarea id="q" placeholder="Ask a question… (Enter to send, Shift+Enter for a new line)"></textarea>
  <button id="send">Send</button>
</footer>
<script>
const log = document.getElementById('log'), q = document.getElementById('q'),
      send = document.getElementById('send'), status = document.getElementById('status');
const CHIPS = __CHIPS__;
const chips = document.getElementById('chips');
CHIPS.forEach(text => {
  const b = document.createElement('div'); b.className = 'chip'; b.textContent = text;
  b.onclick = () => { q.value = text; submit(); }; chips.appendChild(b);
});
function add(role, text) {
  const d = document.createElement('div');
  d.className = 'msg ' + (role === 'user' ? 'user' : 'bot');
  d.innerHTML = '<div class="role">' + (role === 'user' ? 'you' : 'assistant') + '</div>';
  d.appendChild(document.createTextNode(text || '…'));
  log.appendChild(d); log.scrollTop = log.scrollHeight; return d;
}
async function submit() {
  const text = q.value.trim(); if (!text) return;
  q.value = ''; add('user', text);
  const bubble = add('bot', '');
  send.disabled = true;
  try {
    const res = await fetch('/api/chat', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({model: status.dataset.model, stream: false,
                            messages: [{role: 'user', content: text}]})
    });
    const data = await res.json();
    bubble.lastChild.textContent = (data.message && data.message.content) || '(no answer)';
  } catch (e) { bubble.lastChild.textContent = 'Error: ' + e; }
  send.disabled = false; q.focus(); log.scrollTop = log.scrollHeight;
}
send.onclick = submit;
q.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit(); }
});
fetch('/api/tags').then(r => r.json()).then(d => {
  const m = (d.models && d.models[0]) || {};
  status.textContent = (m.name || 'model') + ' · ready'; status.dataset.model = m.name || '';
}).catch(() => { status.textContent = 'model unavailable'; });
</script>
</body>
</html>
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def answer(question: str, max_new_tokens: int = 96) -> str:
    with STATE["lock"]:
        return generate_answer(STATE["model"], STATE["tokenizer"], question, STATE["device"],
                               max_new_tokens=max_new_tokens, temperature=0.2)


class Handler(BaseHTTPRequestHandler):
    server_version = "AirQualitySLM/1.0"

    def log_message(self, fmt: str, *args) -> None:  # quieter logs
        if "--verbose" in sys.argv:
            super().log_message(fmt, *args)

    # ---- helpers ----
    def _json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception:
            return {}

    def _stream_start(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

    def _stream_line(self, payload: Dict[str, Any]) -> None:
        self.wfile.write((json.dumps(payload) + "\n").encode("utf-8"))
        self.wfile.flush()

    # ---- routes ----
    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            chips = STATE.get("questions") or STANDARD_QUESTIONS
            html = INDEX_HTML.replace("__CHIPS__", json.dumps(chips[:4]))
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/api/tags":
            cfg = STATE["config"].to_dict() if STATE["config"] else {}
            self._json({"models": [{
                "name": STATE["name"], "model": STATE["name"],
                "modified_at": _now(), "size": 0,
                "details": {"family": "llama", "parameter_size": f"{cfg.get('n_layer')}L",
                            "quantization_level": "F32"},
            }]})
            return
        if path == "/api/ps":
            self._json({"models": [{"name": STATE["name"], "model": STATE["name"]}]})
            return
        if path == "/health":
            self._json({"status": "ok" if STATE["model"] is not None else "loading"})
            return
        self._json({"error": "not found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]
        body = self._body()

        if path == "/api/chat":
            messages: List[Dict[str, str]] = body.get("messages") or []
            question = ""
            for message in reversed(messages):
                if message.get("role") == "user":
                    question = str(message.get("content", "")).strip()
                    break
            if not question:
                self._json({"error": "no user message"}, status=400)
                return
            started = time.time()
            text = answer(question)
            if body.get("stream"):
                self._stream_start()
                for chunk in _chunks(text):
                    self._stream_line({"model": STATE["name"], "created_at": _now(),
                                       "message": {"role": "assistant", "content": chunk},
                                       "done": False})
                self._stream_line({"model": STATE["name"], "created_at": _now(),
                                   "message": {"role": "assistant", "content": ""},
                                   "done": True, "done_reason": "stop",
                                   "total_duration": int((time.time() - started) * 1e9)})
                return
            self._json({"model": STATE["name"], "created_at": _now(),
                        "message": {"role": "assistant", "content": text},
                        "done": True, "done_reason": "stop",
                        "eval_count": len(text.split()),
                        "total_duration": int((time.time() - started) * 1e9)})
            return

        if path == "/api/generate":
            prompt = str(body.get("prompt", "")).strip()
            if body.get("system"):
                prompt = f"{body['system']}\n{prompt}"
            text = answer(prompt)
            if body.get("stream"):
                self._stream_start()
                for chunk in _chunks(text):
                    self._stream_line({"model": STATE["name"], "created_at": _now(),
                                       "response": chunk, "done": False})
                self._stream_line({"model": STATE["name"], "created_at": _now(),
                                   "response": "", "done": True, "done_reason": "stop"})
                return
            self._json({"model": STATE["name"], "created_at": _now(), "response": text,
                        "done": True, "done_reason": "stop"})
            return

        self._json({"error": "not found"}, status=404)


def _chunks(text: str, size: int = 4) -> List[str]:
    words = text.split(" ")
    return [" ".join(words[i:i + size]) + (" " if i + size < len(words) else "")
            for i in range(0, len(words), size)]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the Air Quality SLM chat demo.")
    parser.add_argument("--models-dir", default=default_models_dir(PROJECT_ROOT))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--name", default=None, help="Model name reported by the API.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        model, tokenizer, config, meta = load_model_for_inference(args.models_dir)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        print("Train the model first:  python src/train_slm.py")
        return 2
    device = torch.device(meta["device"])
    model.to(device)
    if not args.name:
        profile_key = read_model_profile(args.models_dir).get("key", "air_quality")
        args.name = profile_key.replace("_", "-") + "-slm"
    STATE.update(model=model, tokenizer=tokenizer, config=config, meta=meta,
                 device=device, name=args.name,
                 questions=load_questions(args.models_dir))

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"Air Quality SLM chat demo: {url}")
    print(f"  model: {meta['weights_path']} on {device.type}")
    print(f"  Ollama-compatible API: POST {url}/api/chat")
    print("  Ctrl-C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
