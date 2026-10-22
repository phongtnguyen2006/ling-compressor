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
from contextlib import nullcontext
from functools import lru_cache
import json
import math
import mimetypes
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
UI_ROOT = REPO_ROOT / "ui"
SAMPLES_ROOT = REPO_ROOT / "data" / "samples"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from promptcomp import compress  # noqa: E402
from promptcomp.lexical import PilotProvider, VectorIndex, suggest  # noqa: E402
from promptcomp.lexical.tokenization import local_counter  # noqa: E402
from promptcomp.tokens import CachedCounter  # noqa: E402

MAX_INPUT_BYTES = 1_000_000
MAX_REQUEST_BYTES = MAX_INPUT_BYTES * 6 + 16_384
MAX_VECTOR_BYTES = 2_000_000_000
SYNTHETIC_VECTORS = REPO_ROOT / "data" / "lexical-demo" / "synthetic-vectors.txt"
VECTOR_MAX_WORDS = 50_000
_WORDNET_LOCK = Lock()
_VECTOR_LOCK = Lock()
_COUNTER_LOCK = Lock()


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
    examples.append(
        {
            "id": "hybrid_guardrail",
            "name": "Hybrid guardrail",
            "text": "The team worked carefully, unless spending exceeds $5,000.",
            "preview": "See how the protected spending condition survives an LLMLingua-2 pass.",
            "mode": "hybrid_sentinel",
        }
    )
    examples.extend([
        {
            "id": "semantic_phrase",
            "name": "Semantic phrase · B0",
            "text": "They worked in order to finish the assignment.",
            "preview": "A curated phrase proposal with no vector file required.",
            "mode": "semantic",
            "arm": "b0",
        },
        {
            "id": "semantic_protected",
            "name": "Semantic protected text · B1",
            "text": "They must work in order to finish the assignment.",
            "preview": "Inspect why a protected modal sentence stays unchanged.",
            "mode": "semantic",
            "arm": "b1",
            "vector_source": "synthetic",
        },
        {
            "id": "semantic_vectors",
            "name": "Semantic vector proposals · B2",
            "text": "We discuss the methodologies. These projects necessitate planning.",
            "preview": "Inspect B2 proposals using bundled synthetic vectors.",
            "mode": "semantic",
            "arm": "b2",
            "vector_source": "synthetic",
        },
    ])
    return examples


@lru_cache(maxsize=1)
def _hybrid_backend():
    from eval.baselines import LLMLingua2Compressor

    try:
        return LLMLingua2Compressor()
    except ImportError as exc:
        raise RuntimeError(
            'Hybrid mode needs LLMLingua-2. Install it with: python -m pip install -e ".[llmlingua]"'
        ) from exc


@lru_cache(maxsize=4)
def _vector_index(path: str, modified_ns: int, size: int, max_words: int) -> VectorIndex:
    """Cache a local index by resolved file identity and the retention limit."""
    return VectorIndex.from_text(path, max_words=max_words)


@lru_cache(maxsize=2)
def _cached_counter(encoding: str):
    return local_counter(encoding)


def _semantic_counter(encoding: str):
    with _COUNTER_LOCK:
        return _cached_counter(encoding)


def _semantic_payload(payload: dict, text: str) -> dict:
    started = time.perf_counter()
    arm = payload.get("arm", "b0")
    if not isinstance(arm, str) or arm not in {"b0", "b1", "b2"}:
        raise ValueError("Semantic arm must be b0, b1, or b2.")
    use_wordnet = payload.get("use_wordnet", False)
    if not isinstance(use_wordnet, bool):
        raise ValueError("use_wordnet must be a boolean.")
    vector_source = payload.get("vector_source", "synthetic")
    if not isinstance(vector_source, str) or vector_source not in {"synthetic", "local"}:
        raise ValueError("vector_source must be synthetic or local.")
    vector_path = payload.get("vector_path")
    if vector_path is not None and (not isinstance(vector_path, str) or not vector_path.strip()):
        raise ValueError("vector_path must be a nonempty local file path.")
    if vector_path is not None and len(vector_path) > 4096:
        raise ValueError("vector_path is too long.")
    encoding = payload.get("encoding", "o200k_base")
    if not isinstance(encoding, str) or encoding not in {"o200k_base", "cl100k_base"}:
        raise ValueError("encoding must be o200k_base or cl100k_base.")
    max_candidates = payload.get("max_candidates", 8)
    if type(max_candidates) is not int or not 1 <= max_candidates <= 32:
        raise ValueError("max_candidates must be an integer between 1 and 32.")
    default_slots = min(4, max_candidates - 1 if arm == "b2" else max_candidates)
    embedding_slots = payload.get("embedding_slots", default_slots)
    if type(embedding_slots) is not int or not 0 <= embedding_slots <= max_candidates:
        raise ValueError("embedding_slots must be an integer from 0 through max_candidates.")
    if arm == "b2" and not 0 < embedding_slots < max_candidates:
        raise ValueError("B2 requires at least one baseline slot and one embedding slot.")
    threshold = payload.get("min_embedding_similarity")
    if threshold is not None and (
        isinstance(threshold, bool) or not isinstance(threshold, (int, float))
        or not math.isfinite(threshold) or not -1 <= threshold <= 1
    ):
        raise ValueError("min_embedding_similarity must be null or a finite number between -1 and 1.")
    adaptive = payload.get("adaptive_similarity", False)
    if not isinstance(adaptive, bool):
        raise ValueError("adaptive_similarity must be a boolean.")
    if adaptive and threshold is None:
        raise ValueError("adaptive_similarity requires min_embedding_similarity.")
    price = payload.get("input_price_per_million", 0.0)
    if (isinstance(price, bool) or not isinstance(price, (int, float))
            or not math.isfinite(price) or price < 0):
        raise ValueError("input_price_per_million must be a finite nonnegative number.")

    vectors = None
    vector_label = "none"
    if arm != "b0":
        if vector_source == "local" and vector_path is None:
            raise ValueError("B1/B2 with local vectors require vector_path.")
        try:
            path = SYNTHETIC_VECTORS if vector_source == "synthetic" else Path(vector_path).expanduser()
            path = path.resolve()
            if not path.is_file():
                raise ValueError("Vector file does not exist or is not a regular file.")
            stat = path.stat()
            if stat.st_size > MAX_VECTOR_BYTES:
                raise ValueError("Vector file exceeds the 2 GB local workbench limit.")
            with _VECTOR_LOCK:
                vectors = _vector_index(str(path), stat.st_mtime_ns, stat.st_size, VECTOR_MAX_WORDS)
        except (OSError, RuntimeError) as exc:
            raise ValueError("Could not read the local vector file.") from exc
        except ValueError as exc:
            if vector_source == "local" and not str(exc).startswith(("Vector file does not exist", "Vector file exceeds")):
                raise ValueError("Invalid local vector file; check its UTF-8, dimensions, and rows.") from exc
            raise
        vector_label = (
            "Bundled synthetic vectors (software fixture, not trained embeddings)"
            if path == SYNTHETIC_VECTORS.resolve() else str(path)
        )

    # NLTK's lazy WordNet loader is unsafe across concurrent requests.
    with (_WORDNET_LOCK if use_wordnet else nullcontext()):
        provider = PilotProvider(
            arm=arm, vectors=vectors, use_wordnet=use_wordnet,
            max_candidates=max_candidates, embedding_slots=embedding_slots,
        )
        result = suggest(
            text, provider, counter=_semantic_counter(encoding),
            min_embedding_similarity=threshold, adaptive_similarity=adaptive,
        )
    saved = result.tokens_before - result.tokens_after
    timings_ms = {**result.timings_ms, "engine_total": result.timings_ms["total"]}
    timings_ms["total"] = (time.perf_counter() - started) * 1000
    return {
        "mode": "semantic",
        "original": result.original,
        "compressed": result.suggested,
        "tokens_before": result.tokens_before,
        "tokens_after": result.tokens_after,
        "saved_tokens": saved,
        "saved_percent": round(100 * saved / result.tokens_before, 1),
        "ratio": result.tokens_after / result.tokens_before,
        "timings_ms": timings_ms,
        "edits": [dataclasses.asdict(edit) for edit in result.edits],
        "decisions": [dataclasses.asdict(decision) for decision in result.decisions],
        "warnings": list(result.warnings),
        "review_status": result.review_status,
        "token_counter": result.token_counter,
        "config_hash": result.config_hash,
        "provider": result.provider,
        "provider_metadata": provider.metadata,
        "vector_label": vector_label,
        "input_price_per_million": float(price),
        "cost_saved_usd": saved * price / 1_000_000,
    }


def _compress_payload(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Enter some text to compress.")
    if len(text.encode("utf-8")) > MAX_INPUT_BYTES:
        raise ValueError("Input is too large. Keep it under 1 MB.")

    mode = payload.get("mode", "conservative")
    if mode == "semantic":
        return _semantic_payload(payload, text)

    try:
        target_ratio = float(payload.get("target_ratio", 0.6))
    except (TypeError, ValueError) as exc:
        raise ValueError("Target ratio must be a number.") from exc
    if not 0.1 <= target_ratio <= 1.0:
        raise ValueError("Target ratio must be between 0.1 and 1.0.")

    if not isinstance(mode, str) or mode not in {"conservative", "hybrid_sentinel"}:
        raise ValueError("Choose tree pruning, guarded hybrid, or semantic substitution.")

    if mode == "hybrid_sentinel":
        from eval.hybrid import SentinelHybridCompressor

        hybrid = SentinelHybridCompressor(ll_compressor=_hybrid_backend())
        output = hybrid.compress(text, target_ratio)
        counter = _semantic_counter("o200k_base")
        before = counter.count(text)
        after = counter.count(output.text)
        return {
            "mode": mode,
            "original": text,
            "compressed": output.text,
            "tokens_before": before,
            "tokens_after": after,
            "saved_tokens": before - after,
            "saved_percent": round((1 - after / before) * 100, 1),
            "ratio": after / before,
            "timings_ms": {"total": output.latency_ms},
            "target_ratio": target_ratio,
            "hybrid_stats": hybrid.stats,
        }

    result = compress(
        text,
        target_ratio=target_ratio,
        min_tokens=0,
        substitute=bool(payload.get("substitute", True)),
        counter=CachedCounter(_semantic_counter("o200k_base"), REPO_ROOT / ".token_cache.sqlite"),
    )
    data = dataclasses.asdict(result)
    data["saved_tokens"] = result.tokens_before - result.tokens_after
    data["saved_percent"] = round((1 - result.ratio) * 100, 1)
    data["target_ratio"] = target_ratio
    data["mode"] = mode
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
        origin = self.headers.get("Origin")
        if origin:
            parsed_origin = urlparse(origin)
            if parsed_origin.scheme not in {"http", "https"} or parsed_origin.netloc != self.headers.get("Host"):
                self._send_json({"error": "Cross-origin requests are not allowed."}, HTTPStatus.FORBIDDEN)
                return
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            self._send_json({"error": "Send JSON with Content-Type: application/json."}, HTTPStatus.UNSUPPORTED_MEDIA_TYPE)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 0:
                raise ValueError("Content-Length must be nonnegative.")
            if length > MAX_REQUEST_BYTES:
                raise ValueError("Request body is too large. Keep text under 1 MB.")
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
