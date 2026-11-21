"""Browser UI for the NovaMart support agent (standard library only).

    python -m novamart.web                 # http://127.0.0.1:8000
    python -m novamart.web --port 8080     # another port

Serves one HTML page (novamart/static/index.html) and a tiny JSON API:
    GET  /api/info    model, today, mock orders, policy sections
    POST /api/ask     {"session": "...", "question": "..."} -> answer + tool trace
    POST /api/reset   {"session": "..."} -> clears that conversation

Each browser tab gets its own conversation (SupportAgent); all of them share
one Gemini client and one policy index, which is built once at startup.
"""

from __future__ import annotations

import argparse
import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from novamart import config
from novamart.agent import SupportAgent, make_client
from novamart.quota import QuotaExhaustedError
from novamart.retrieval import build_index
from novamart.tools import Toolbox

STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_SESSIONS = 200

EXAMPLES = [
    "Can a customer return an opened laptop?",
    "Where is order NM-10002?",
    "The customer on NM-10001 wants to return their opened laptop today. Eligible?",
    "Does the employee discount apply to electronics?",
    "Do we match Amazon's prices?",
    "What's our policy on dogs in the store?",
]


class App:
    """Shared state: one client + index, one agent per browser session."""

    def __init__(self) -> None:
        print("Building policy index (first run embeds the documents)...", flush=True)
        self.client = make_client()
        self.toolbox = Toolbox(build_index(self.client))
        self.sessions: dict[str, tuple[SupportAgent, threading.Lock]] = {}
        self.lock = threading.Lock()
        self.quota_message: str | None = None  # set while the daily quota is used up

    def session(self, sid: str) -> tuple[SupportAgent, threading.Lock]:
        with self.lock:
            if sid not in self.sessions:
                if len(self.sessions) >= MAX_SESSIONS:  # drop the oldest
                    self.sessions.pop(next(iter(self.sessions)))
                self.sessions[sid] = (SupportAgent(self.client, self.toolbox), threading.Lock())
            return self.sessions[sid]

    def info(self) -> dict:
        return {
            "model": config.CHAT_MODEL,
            "today": config.today().isoformat(),
            "min_score": config.MIN_SCORE,
            "quota_message": self.quota_message,
            "examples": EXAMPLES,
            "orders": [
                {
                    "id": oid,
                    "status": o.get("status"),
                    "items": ", ".join(i["name"] for i in o.get("items", [])),
                }
                for oid, o in self.toolbox.orders.items()
            ],
            "policies": [
                {"source": c.id, "document": c.doc_title, "section": c.section}
                for c in self.toolbox.index.chunks
            ],
        }

    def ask(self, sid: str, question: str) -> dict:
        agent, lock = self.session(sid)
        with lock:  # one question at a time per conversation
            turn = agent.ask(question)
        retrieved = turn.retrieved_sources
        return {
            "answer": turn.answer,
            "refused": turn.refused,
            "cited": sorted(turn.cited_sources),
            "unsupported_citations": sorted(turn.cited_sources - retrieved),
            "tool_calls": [
                {"name": c.name, "args": c.args, "result": c.result} for c in turn.tool_calls
            ],
        }

    def reset(self, sid: str) -> None:
        with self.lock:
            self.sessions.pop(sid, None)


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload: dict, status: int = 200) -> None:
            self._send(status, json.dumps(payload).encode(), "application/json")

        def _body(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            if length > 20_000:
                raise ValueError("request too large")
            return json.loads(self.rfile.read(length) or b"{}")

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                html = (STATIC_DIR / "index.html").read_bytes()
                self._send(200, html, "text/html; charset=utf-8")
            elif self.path == "/api/info":
                self._json(app.info())
            else:
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            try:
                body = self._body()
                sid = str(body.get("session", ""))[:64] or "default"
                if self.path == "/api/ask":
                    question = str(body.get("question", "")).strip()[:2000]
                    if not question:
                        return self._json({"error": "empty question"}, HTTPStatus.BAD_REQUEST)
                    self._json(app.ask(sid, question))
                    app.quota_message = None  # a successful answer clears the banner
                elif self.path == "/api/reset":
                    app.reset(sid)
                    self._json({"ok": True})
                else:
                    self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except QuotaExhaustedError as exc:
                app.quota_message = str(exc)
                print(f"!!! QUOTA EXHAUSTED: {exc}", flush=True)
                self._json({"error": str(exc), "quota_exhausted": True}, HTTPStatus.TOO_MANY_REQUESTS)
            except Exception as exc:  # surface other API errors in the UI
                self._json({"error": f"{type(exc).__name__}: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

        def log_message(self, fmt: str, *args) -> None:
            if self.path.startswith("/api/ask"):
                super().log_message(fmt, *args)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="NovaMart support agent web UI")
    parser.add_argument("--host", default="127.0.0.1", help="default 127.0.0.1 (this machine only)")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    try:
        app = App()  # may embed the policy documents
    except QuotaExhaustedError as exc:
        raise SystemExit(f"!!! QUOTA EXHAUSTED: {exc}")
    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    print(f"NovaMart UI running at http://{args.host}:{args.port}  (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
