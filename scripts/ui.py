"""Local web workbench for inspecting promptcomp compression.

Run from the repository root:
    python scripts/ui.py
"""
from __future__ import annotations

import argparse
import os
import sys

# scripts/inspect.py would otherwise shadow the stdlib inspect module when this
# file is launched directly, which breaks dataclasses during import.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [path for path in sys.path if os.path.abspath(path or ".") != _HERE]

import dataclasses
import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
UI_ROOT = REPO_ROOT / "ui"
SAMPLES_ROOT = REPO_ROOT / "data" / "samples"
sys.path.insert(0, str(REPO_ROOT / "src"))

from promptcomp import compress  # noqa: E402

MAX_INPUT_BYTES = 1_000_000


def _example_catalog() -> list[dict[str, str]]:
    labels = {
        "memo1": "Expense policy memo",
        "notice_snip": "Formal notice",
        "policy_snip": "Reimbursement policy",
    }
    examples = []
    for path in sorted(SAMPLES_ROOT.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        examples.append(
            {
                "id": path.stem,
                "name": labels.get(path.stem, path.stem.replace("_", " ").title()),
                "text": text,
                "preview": " ".join(text.split())[:150],
            }
        )
    return examples


def _compress_payload(payload: dict) -> dict:
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Enter some text to compress.")
    if len(text.encode("utf-8")) > MAX_INPUT_BYTES:
        raise ValueError("Input is too large. Keep it under 1 MB.")

    try:
        target_ratio = float(payload.get("target_ratio", 0.6))
    except (TypeError, ValueError) as exc:
        raise ValueError("Target ratio must be a number.") from exc
    if not 0.1 <= target_ratio <= 1.0:
        raise ValueError("Target ratio must be between 0.1 and 1.0.")

    result = compress(
        text,
        target_ratio=target_ratio,
        min_tokens=0,
        substitute=bool(payload.get("substitute", True)),
        cache_path=REPO_ROOT / ".token_cache.sqlite",
    )
    data = dataclasses.asdict(result)
    data["saved_tokens"] = result.tokens_before - result.tokens_after
    data["saved_percent"] = round((1 - result.ratio) * 100, 1)
    data["target_ratio"] = target_ratio
    return data


class WorkbenchHandler(BaseHTTPRequestHandler):
    server_version = "PromptcompWorkbench/1.0"

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/examples":
            self._send_json({"examples": _example_catalog()})
            return
        if path == "/health":
            self._send_json({"status": "ok"})
            return
        self._serve_static(path)

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/compress":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > MAX_INPUT_BYTES:
                raise ValueError("Input is too large. Keep it under 1 MB.")
            payload = json.loads(self.rfile.read(length) or b"{}")
            self._send_json(_compress_payload(payload))
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:  # keep the local UI useful when optional models are missing
            self._send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def _serve_static(self, request_path: str) -> None:
        relative = "index.html" if request_path in {"", "/"} else request_path.lstrip("/")
        candidate = (UI_ROOT / relative).resolve()
        if UI_ROOT.resolve() not in candidate.parents and candidate != UI_ROOT.resolve():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if not candidate.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        body = candidate.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        print(f"[workbench] {self.address_string()} {format % args}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the promptcomp web workbench.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    server = ThreadingHTTPServer((args.host, args.port), WorkbenchHandler)
    print(f"Ling Compressor Workbench running at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping workbench.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
