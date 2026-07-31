"""Tiny OpenAI-compatible mock server for plumbing tests (no GPU)."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class _Handler(BaseHTTPRequestHandler):
    server_version = "catan-mock/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        return

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            body = {}
        # Intentionally invalid / non-legal JSON so first_legal path is exercised
        # unless the client asks for json_object — then return a stub action.
        structured = isinstance(body.get("response_format"), dict)
        content = '{"action": 0, "rationale": "mock"}' if structured else "not-json"
        payload = {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "model": body.get("model", "mock"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
        }
        data = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def serve_forever(host: str = "127.0.0.1", port: int = 8000) -> None:
    httpd = ThreadingHTTPServer((host, port), _Handler)
    print(f"mock OpenAI server on http://{host}:{port}/v1", flush=True)
    httpd.serve_forever()
