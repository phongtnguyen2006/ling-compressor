"""Local web workbench for inspecting promptcomp compression.

Security Notice:
Semantic template persistence, preview, validation, and listing endpoints
(/api/semantic/*) are strictly restricted to local loopback connections
(127.0.0.1 / localhost) and local server bindings. This prevents unintended
network exposure of private prompt templates. This prototype does not claim
network production server readiness; use in multi-user or network-exposed
environments requires authentication and access controls.

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

import dataclasses  # noqa: E402
import ipaddress  # noqa: E402
from contextlib import nullcontext  # noqa: E402
from functools import lru_cache  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import mimetypes  # noqa: E402
import time  # noqa: E402
from http import HTTPStatus  # noqa: E402
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer  # noqa: E402
from pathlib import Path  # noqa: E402
from threading import Lock  # noqa: E402
from urllib.parse import urlparse  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
UI_ROOT = REPO_ROOT / "ui"
SAMPLES_ROOT = REPO_ROOT / "data" / "samples"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from promptcomp import compress  # noqa: E402
from promptcomp.lexical import PilotProvider, VectorIndex, suggest  # noqa: E402
from promptcomp.lexical.approval import (  # noqa: E402
    OccurrenceReview,
    TemplateStore,
    WORKFLOW_DISCLAIMER,
    create_approved_template,
    preview_approval,
    validate_and_apply,
)
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
_TEMPLATE_STORE = TemplateStore(REPO_ROOT / ".cache" / "approved_templates")


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
            "name": "Authored prompt template · B0",
            "text": "Hello {customer_name}, please review account details in order to confirm order #{order_id}.",
            "preview": "Authored template demo with protected placeholders and a reviewable phrase edit.",
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


def _normalize_encoding(encoding: object) -> str:
    """Normalize tokenizer encoding string (e.g. 'tiktoken:o200k_base' -> 'o200k_base')."""
    if encoding is None:
        return "o200k_base"
    if not isinstance(encoding, str) or not encoding.strip():
        raise ValueError("encoding must be o200k_base or cl100k_base.")
    clean = encoding.strip().removeprefix("tiktoken:")
    if clean in {"o200k_base", "cl100k_base"}:
        return clean
    raise ValueError("encoding must be o200k_base or cl100k_base.")


def _build_semantic_provider_and_label(payload: dict) -> tuple[PilotProvider, str, float | None, bool, str]:
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
    encoding = _normalize_encoding(payload.get("encoding", "o200k_base"))
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

    provider = PilotProvider(
        arm=arm, vectors=vectors, use_wordnet=use_wordnet,
        max_candidates=max_candidates, embedding_slots=embedding_slots,
    )
    return provider, vector_label, threshold, adaptive, encoding


def _semantic_payload(payload: dict, text: str) -> dict:
    started = time.perf_counter()
    price = payload.get("input_price_per_million", 0.0)
    if (isinstance(price, bool) or not isinstance(price, (int, float))
            or not math.isfinite(price) or price < 0):
        raise ValueError("input_price_per_million must be a finite nonnegative number.")

    provider, vector_label, threshold, adaptive, encoding = _build_semantic_provider_and_label(payload)
    use_wordnet = payload.get("use_wordnet", False)

    # NLTK's lazy WordNet loader is unsafe across concurrent requests.
    with (_WORDNET_LOCK if use_wordnet else nullcontext()):
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
        "workflow_disclaimer": WORKFLOW_DISCLAIMER,
        "occurrences": [
            {
                "span": {"start": edit.span.start, "end": edit.span.end},
                "original": edit.original,
                "replacement": edit.replacement,
                "decision": "pending",
                "provenance": edit.provenance,
                "similarity": edit.similarity,
                "token_delta": edit.token_delta,
                "pos": edit.pos,
                "sentence_before": edit.sentence_before,
                "sentence_after": edit.sentence_after,
            }
            for edit in result.edits
        ],
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


def _is_loopback(address: str | None) -> bool:
    """Return True if the IP address belongs to a local loopback interface."""
    if not address:
        return False
    clean = address.strip("[]").lower()
    if clean == "localhost":
        return True
    try:
        ip = ipaddress.ip_address(clean)
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return ip.is_loopback


def _allowed_semantic_host(host_header: str | None, server_port: int) -> bool:
    """Accept only literal loopback IPs or localhost in semantic Host headers."""
    if not host_header or any(char in host_header for char in "/\\@?# \t\r\n"):
        return False
    try:
        parsed = urlparse("//" + host_header)
        return (parsed.netloc == host_header and _is_loopback(parsed.hostname)
                and parsed.port in {None, server_port})
    except ValueError:
        return False


def _semantic_preview_payload(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Enter some text to preview.")
    if len(text.encode("utf-8")) > MAX_INPUT_BYTES:
        raise ValueError("Input is too large. Keep text under 1 MB.")
    decisions = payload.get("decisions", [])
    if not isinstance(decisions, list):
        raise ValueError("decisions must be a list.")
    encoding = _normalize_encoding(payload.get("encoding", "o200k_base"))
    config_hash = str(payload.get("config_hash", ""))
    vector_identity = str(payload.get("vector_identity", "none"))
    explicit_protected = payload.get("protected_spans")

    return preview_approval(
        text,
        decisions,
        config_hash=config_hash,
        tokenizer_name=encoding,
        vector_identity=vector_identity,
        counter=_semantic_counter(encoding),
        explicit_protected_spans=explicit_protected,
    )


def _semantic_approve_payload(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Enter some text to approve.")
    if len(text.encode("utf-8")) > MAX_INPUT_BYTES:
        raise ValueError("Input is too large. Keep text under 1 MB.")
    decisions = payload.get("decisions", [])
    if not isinstance(decisions, list):
        raise ValueError("decisions must be a list.")
    reviewer = str(payload.get("reviewer", "local-user"))
    template_id = payload.get("template_id")
    if template_id is not None and not isinstance(template_id, str):
        raise ValueError("template_id must be a string if provided.")
    explicit_protected = payload.get("protected_spans")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a JSON object.")

    # Server-side recomputation and verification of suggestion and configuration
    provider, server_vector_label, threshold, adaptive, encoding = _build_semantic_provider_and_label(payload)
    use_wordnet = payload.get("use_wordnet", False)
    with (_WORDNET_LOCK if use_wordnet else nullcontext()):
        server_res = suggest(
            text,
            provider,
            counter=_semantic_counter(encoding),
            min_embedding_similarity=threshold,
            adaptive_similarity=adaptive,
            protected_spans=explicit_protected or (),
        )
    server_config_hash = server_res.config_hash

    # Verify client config_hash if provided
    client_config_hash = payload.get("config_hash")
    if client_config_hash and str(client_config_hash) != server_config_hash:
        raise ValueError(
            f"Configuration hash mismatch: provided '{client_config_hash}' "
            f"does not match server-generated '{server_config_hash}'."
        )

    # Verify client vector_identity if provided
    client_vector_identity = payload.get("vector_identity")
    if client_vector_identity and str(client_vector_identity) != server_vector_label:
        raise ValueError(
            f"Vector identity mismatch: provided '{client_vector_identity}' "
            f"does not match server-resolved '{server_vector_label}'."
        )

    # Only selected edits are eligible; rejected candidates are audit records.
    selected = {(e.span.start, e.span.end, e.original, e.replacement): e for e in server_res.edits}
    if not selected:
        raise ValueError("No token-saving candidate suggestions available to approve.")
    reviewed = {}
    for d in decisions:
        if not isinstance(d, dict):
            raise ValueError("Each decision must be a dictionary.")
        raw_span = d.get("span")
        if not isinstance(raw_span, dict) or type(raw_span.get("start")) is not int or type(raw_span.get("end")) is not int:
            raise ValueError(f"Invalid span in decision: {raw_span!r}")
        key = (raw_span["start"], raw_span["end"], d.get("original"), d.get("replacement"))
        if key not in selected:
            raise ValueError(
                f"Unauthorized substitution '{d.get('original')}' -> '{d.get('replacement')}' "
                f"at span [{raw_span['start']}:{raw_span['end']}]. "
                "All approval decisions must correspond to server-selected edits."
            )
        if key in reviewed:
            raise ValueError("Duplicate approval decision for a server-selected edit.")
        if d.get("decision") not in {"accepted", "rejected"}:
            raise ValueError("Every selected edit must be explicitly accepted or rejected before saving.")
        reviewed[key] = d["decision"]
    if len(reviewed) != len(selected):
        raise ValueError("Every server-selected edit needs exactly one approval decision; some are missing.")

    canonical_decisions = [
        OccurrenceReview(
            span=edit.span, original=edit.original, replacement=edit.replacement,
            decision=reviewed[(edit.span.start, edit.span.end, edit.original, edit.replacement)],
            provenance=edit.provenance, similarity=edit.similarity,
            token_delta=edit.token_delta, pos=edit.pos,
            sentence_before=edit.sentence_before, sentence_after=edit.sentence_after,
        )
        for edit in server_res.edits
    ]

    template = create_approved_template(
        text,
        canonical_decisions,
        config_hash=server_config_hash,
        tokenizer_name=encoding,
        vector_identity=server_vector_label,
        counter=_semantic_counter(encoding),
        reviewer=reviewer,
        template_id=template_id,
        explicit_protected_spans=explicit_protected,
        metadata=metadata,
    )
    try:
        saved_path = _TEMPLATE_STORE.save(template)
    except FileExistsError as exc:
        raise ValueError(str(exc)) from exc

    return {
        "status": "saved",
        "path": str(saved_path),
        "template": template.to_dict(),
        "audit": template.to_audit_dict(),
    }


def _semantic_validate_payload(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")
    text = payload.get("text")
    if not isinstance(text, str):
        raise ValueError("text must be a string.")
    if len(text.encode("utf-8")) > MAX_INPUT_BYTES:
        raise ValueError("Input is too large. Keep text under 1 MB.")

    template_id = payload.get("template_id")
    if not template_id or not isinstance(template_id, str):
        raise ValueError("template_id is required to validate an approved template.")
    template = _TEMPLATE_STORE.get(template_id)
    if template is None:
        raise ValueError(f"Approved template '{template_id}' not found in store.")

    if "template" in payload or "config_hash" in payload or "vector_identity" in payload:
        raise ValueError(
            "Validation accepts a saved template_id and active semantic settings only. "
            "Hash-only replay is rejected."
        )

    explicit_protected = payload.get("protected_spans")

    # Determine active configuration identities (fail-closed: require active semantic config; reject hash-only replay)
    if "arm" not in payload:
        raise ValueError(
            "Active semantic configuration ('arm', etc.) is required for live validation. "
            "Hash-only replay is rejected."
        )

    provider, active_vector_identity, threshold, adaptive, encoding = _build_semantic_provider_and_label(payload)
    use_wordnet = payload.get("use_wordnet", False)
    with (_WORDNET_LOCK if use_wordnet else nullcontext()):
        server_res = suggest(
            text,
            provider,
            counter=_semantic_counter(encoding),
            min_embedding_similarity=threshold,
            adaptive_similarity=adaptive,
            protected_spans=explicit_protected or (),
        )
    active_config_hash = server_res.config_hash

    # Recompute candidate validity/provenance: verify that live active engine still produces accepted substitutions
    live_selected = {(e.span.start, e.span.end, e.original, e.replacement): e for e in server_res.edits}
    for d in template.decisions:
        if d.decision == "accepted":
            key = (d.span.start, d.span.end, d.original, d.replacement)
            if key not in live_selected:
                return {
                    "is_valid": False,
                    "status": "invalidated",
                    "text": text,
                    "invalidation_reason": (
                        f"Active live configuration no longer produces accepted substitution "
                        f"'{d.original}' -> '{d.replacement}' at span [{d.span.start}:{d.span.end}]."
                    ),
                    "tokens_before": server_res.tokens_before,
                    "tokens_after": server_res.tokens_before,
                    "tokens_saved": 0,
                    "binding": template.binding.to_dict(),
                    "warnings": ["Live active engine invalidated accepted template substitution."],
                }

    result = validate_and_apply(
        template,
        text,
        config_hash=active_config_hash,
        tokenizer_name=encoding,
        vector_identity=active_vector_identity,
        counter=_semantic_counter(encoding),
        explicit_protected_spans=explicit_protected,
    )
    return result.to_dict()


def _semantic_list_templates() -> dict:
    templates = _TEMPLATE_STORE.list_all()
    return {"templates": [t.to_dict() for t in templates]}


class WorkbenchHandler(BaseHTTPRequestHandler):
    server_version = "PromptcompWorkbench/1.0"

    def _check_loopback_access(self, req_path: str) -> bool:
        """Reject non-loopback bind or requests for private semantic routes."""
        if not req_path.startswith("/api/semantic/"):
            return True
        server_host = getattr(self.server, "server_address", ("", 0))[0]
        client_host = self.client_address[0] if self.client_address else ""
        server_port = getattr(self.server, "server_address", ("", 0))[1]
        if (
            not _is_loopback(server_host)
            or not _is_loopback(client_host)
            or not _allowed_semantic_host(self.headers.get("Host"), server_port)
        ):
            self._send_json(
                {
                    "error": (
                        "Access forbidden: semantic template and approval endpoints are restricted "
                        "to local loopback access only (local host binding and 127.0.0.1 client) "
                        "to protect prompt confidentiality."
                    )
                },
                HTTPStatus.FORBIDDEN,
            )
            return False
        return True

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if not self._check_loopback_access(path):
            return
        if path == "/api/examples":
            self._send_json({"examples": _example_catalog()})
            return
        if path == "/health":
            self._send_json({"status": "ok"})
            return
        if path == "/api/semantic/templates":
            self._send_json(_semantic_list_templates())
            return
        self._serve_static(path)

    def do_POST(self) -> None:  # noqa: N802
        req_path = urlparse(self.path).path
        if req_path not in {
            "/api/compress",
            "/api/semantic/preview",
            "/api/semantic/approve",
            "/api/semantic/validate",
        }:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if not self._check_loopback_access(req_path):
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
            if req_path == "/api/compress":
                self._send_json(_compress_payload(payload))
            elif req_path == "/api/semantic/preview":
                self._send_json(_semantic_preview_payload(payload))
            elif req_path == "/api/semantic/approve":
                self._send_json(_semantic_approve_payload(payload))
            elif req_path == "/api/semantic/validate":
                self._send_json(_semantic_validate_payload(payload))
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
