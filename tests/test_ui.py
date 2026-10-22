from importlib.util import module_from_spec, spec_from_file_location
import http.client
import json
from pathlib import Path
import subprocess
import sys
from threading import Thread

import pytest

from eval.baselines import Compressed

ui_path = Path(__file__).resolve().parents[1] / "scripts" / "ui.py"
spec = spec_from_file_location("ui_server", ui_path)
ui_server = module_from_spec(spec)
spec.loader.exec_module(ui_server)


def test_example_catalog_contains_sample_documents():
    examples = ui_server._example_catalog()
    assert {item["id"] for item in examples} >= {"memo1", "notice_snip", "policy_snip", "hybrid_guardrail"}
    assert all(item["text"] and item["preview"] for item in examples)
    assert next(item for item in examples if item["id"] == "hybrid_guardrail")["mode"] == "hybrid_sentinel"
    semantic = [item for item in examples if item.get("mode") == "semantic"]
    assert len(semantic) == 3
    assert {item["arm"] for item in semantic} == {"b0", "b1", "b2"}


def test_ui_server_runs_as_documented_command():
    proc = subprocess.run(
        [sys.executable, str(ui_path), "--help"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "web workbench" in proc.stdout.lower()


def test_ui_contains_separate_system_diagram_view():
    html = (ui_server.UI_ROOT / "index.html").read_text()
    assert 'role="tablist"' in html
    assert 'id="system-view"' in html
    assert "Compression data plane" in html
    assert "protected_before == protected_after" in html


def test_compress_payload_supports_single_sentence(tmp_path, monkeypatch):
    monkeypatch.setattr(ui_server, "REPO_ROOT", tmp_path)
    data = ui_server._compress_payload(
        {
            "text": "The committee approved the budget in order to begin quickly.",
            "target_ratio": 0.6,
            "substitute": True,
        }
    )
    assert data["tokens_after"] <= data["tokens_before"]
    assert "in order to" not in data["compressed"]
    assert data["target_ratio"] == 0.6
    assert data["mode"] == "conservative"


def test_hybrid_payload_restores_protected_clause(monkeypatch):
    class FakeLLMLingua:
        def compress_forced(self, text, ratio, force_tokens):
            assert "KEEPCLAUSEA" in force_tokens
            return Compressed(text="The team worked KEEPCLAUSEA.", latency_ms=1.0)

    monkeypatch.setattr(ui_server, "_hybrid_backend", lambda: FakeLLMLingua())
    data = ui_server._compress_payload(
        {"text": "The team worked carefully, unless spending exceeds $5,000.", "mode": "hybrid_sentinel"}
    )
    assert data["compressed"] == "The team worked unless spending exceeds $5,000."
    assert data["hybrid_stats"]["masked_sents"] == 1
    assert data["tokens_after"] < data["tokens_before"]


def test_semantic_phrase_edit_returns_auditable_counts_and_estimate():
    text = "They worked in order to finish the assignment."
    data = ui_server._compress_payload({
        "mode": "semantic", "text": text, "arm": "b0",
        "vector_source": "local", "input_price_per_million": 2.5,
    })
    assert data["mode"] == "semantic"
    assert data["original"] == text
    assert data["compressed"] == "They worked to finish the assignment."
    assert data["saved_tokens"] == data["tokens_before"] - data["tokens_after"] > 0
    assert data["ratio"] == data["tokens_after"] / data["tokens_before"]
    assert data["saved_percent"] == round(100 * data["saved_tokens"] / data["tokens_before"], 1)
    assert data["cost_saved_usd"] == data["saved_tokens"] * 2.5 / 1_000_000
    assert data["vector_label"] == "none"
    assert data["provider_metadata"]["arm"] == "b0"
    assert data["review_status"] == "pending"
    assert data["token_counter"]
    assert data["timings_ms"]["total"] >= 0
    assert data["timings_ms"]["total"] >= data["timings_ms"]["engine_total"]
    assert len(data["config_hash"]) == 64
    assert data["provider"]
    assert data["edits"][0]["original"] == "in order to"
    assert data["edits"][0]["replacement"] == "to"
    assert data["decisions"][0]["decision"] == "selected"
    assert data["warnings"]


def test_semantic_protected_sentence_is_unchanged():
    text = "They must work in order to finish the assignment."
    data = ui_server._compress_payload({"mode": "semantic", "text": text, "arm": "b0"})
    assert data["compressed"] == text
    assert data["tokens_before"] == data["tokens_after"]
    assert data["saved_tokens"] == data["cost_saved_usd"] == 0
    assert data["edits"] == []
    assert data["decisions"][0]["reason"].startswith("protected_")


@pytest.mark.parametrize("arm,budget,slots", [
    ("b0", 1, 1), ("b0", 2, 2), ("b1", 3, 3), ("b2", 2, 1), ("b2", 3, 2),
])
def test_semantic_small_candidate_budgets_use_valid_default_slots(arm, budget, slots):
    data = ui_server._compress_payload({
        "mode": "semantic", "text": "They worked in order to finish the assignment.",
        "arm": arm, "max_candidates": budget,
    })
    assert data["provider_metadata"]["embedding_slots"] == slots


def test_semantic_local_vector_path_errors_are_user_facing(tmp_path):
    loop = tmp_path / "loop"
    loop.symlink_to(loop)
    with pytest.raises(ValueError, match="Could not read the local vector file"):
        ui_server._compress_payload({
            "mode": "semantic", "text": "Write a short answer.", "arm": "b1",
            "vector_source": "local", "vector_path": str(loop),
        })


def test_semantic_b2_uses_bundled_synthetic_vectors():
    data = ui_server._compress_payload({
        "mode": "semantic", "text": "We discuss the methodologies. These projects necessitate planning.",
        "arm": "b2", "vector_source": "synthetic", "max_candidates": 8,
        "embedding_slots": 4, "min_embedding_similarity": 0.75,
        "adaptive_similarity": True,
    })
    assert data["provider_metadata"]["arm"] == "b2"
    assert data["provider_metadata"]["vectors"]["max_words"] == 50_000
    assert "synthetic" in data["vector_label"].lower()
    assert data["decisions"]
    assert data["review_status"] == "pending"


def test_semantic_local_vectors_are_cached_by_file_metadata(tmp_path):
    source = ui_server.SYNTHETIC_VECTORS.read_text(encoding="utf-8")
    path = tmp_path / "vectors.txt"
    path.write_text(source, encoding="utf-8")
    payload = {"mode": "semantic", "text": "Write a short answer.", "arm": "b1",
               "vector_source": "local", "vector_path": str(path)}
    ui_server._vector_index.cache_clear()
    first = ui_server._compress_payload(payload)
    ui_server._compress_payload(payload)
    assert ui_server._vector_index.cache_info().hits == 1
    assert first["vector_label"] == str(path.resolve())
    path.write_text(source.replace("12 4\n", "13 4\n", 1) + "extra 0 0 0 1\n", encoding="utf-8")
    changed = ui_server._compress_payload(payload)
    assert ui_server._vector_index.cache_info().misses == 2
    assert changed["provider_metadata"]["vectors"]["source_rows"] == 13


@pytest.mark.parametrize("settings, message", [
    ({"arm": "b3"}, "arm"),
    ({"arm": ["b0"]}, "arm"),
    ({"use_wordnet": "false"}, "use_wordnet"),
    ({"vector_source": "remote"}, "vector_source"),
    ({"vector_source": ["synthetic"]}, "vector_source"),
    ({"arm": "b1", "vector_source": "local"}, "vector_path"),
    ({"arm": "b2", "vector_source": "local", "vector_path": "/no/such/vectors.txt"}, "Vector file"),
    ({"encoding": "invalid"}, "encoding"),
    ({"encoding": ["o200k_base"]}, "encoding"),
    ({"max_candidates": True}, "max_candidates"),
    ({"embedding_slots": -1}, "embedding_slots"),
    ({"arm": "b2", "max_candidates": 2, "embedding_slots": 2}, "B2"),
    ({"min_embedding_similarity": 1.1}, "min_embedding_similarity"),
    ({"adaptive_similarity": True}, "min_embedding_similarity"),
    ({"input_price_per_million": -1}, "input_price_per_million"),
    ({"input_price_per_million": float("nan")}, "input_price_per_million"),
])
def test_semantic_rejects_bad_settings(settings, message):
    with pytest.raises(ValueError, match=message):
        ui_server._compress_payload({"text": "Write a short answer.", "mode": "semantic", **settings})


def test_semantic_rejects_oversized_text():
    with pytest.raises(ValueError, match="too large"):
        ui_server._compress_payload({"text": "x" * (ui_server.MAX_INPUT_BYTES + 1), "mode": "semantic"})


def test_compress_payload_requires_json_object():
    with pytest.raises(ValueError, match="JSON object"):
        ui_server._compress_payload([{"text": "hello"}])


def test_semantic_http_routes_return_examples_result_and_validation_error():
    server = ui_server.ThreadingHTTPServer(("127.0.0.1", 0), ui_server.WorkbenchHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = http.client.HTTPConnection(*server.server_address, timeout=10)
    try:
        connection.request("GET", "/api/examples")
        response = connection.getresponse()
        examples = json.loads(response.read())["examples"]
        assert response.status == 200
        assert len([item for item in examples if item.get("mode") == "semantic"]) == 3

        connection.request("POST", "/api/compress", body=json.dumps({
            "mode": "semantic", "arm": "b0",
            "text": "They worked in order to finish the assignment.",
        }), headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        result = json.loads(response.read())
        assert response.status == 200
        assert result["compressed"] == "They worked to finish the assignment."

        connection.request("POST", "/api/compress", body=json.dumps({
            "mode": "semantic", "arm": "b1", "vector_source": "local",
            "text": "Write a short answer.",
        }), headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        error = json.loads(response.read())
        assert response.status == 400
        assert "vector_path" in error["error"]

        body = json.dumps({"mode": "semantic", "text": "Write a short answer."})
        connection.request("POST", "/api/compress", body=body, headers={
            "Content-Type": "text/plain", "Origin": "https://elsewhere.example",
        })
        response = connection.getresponse()
        assert response.status == 403
        response.read()

        connection.request("POST", "/api/compress", body=body, headers={"Content-Type": "text/plain"})
        response = connection.getresponse()
        assert response.status == 415
        response.read()
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


@pytest.mark.parametrize("payload", [{}, {"text": "  "}, {"text": "hello", "target_ratio": 2}, {"text": "hello", "mode": "unknown"}])
def test_compress_payload_rejects_invalid_input(payload):
    with pytest.raises(ValueError):
        ui_server._compress_payload(payload)
