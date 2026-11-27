from importlib.util import module_from_spec, spec_from_file_location
import dataclasses
import http.client
import json
from pathlib import Path
import subprocess
import sys
from threading import Thread

import pytest

from eval.baselines import Compressed
from promptcomp.types import Deletion, Span

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
    combined = next(item for item in examples if item["id"] == "tree_semantic_stages")
    assert combined["mode"] == "tree_semantic"
    assert combined["target_ratio"] == 0.9
    assert "in order to" in combined["text"]


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


def test_tree_semantic_reports_distinct_stage_gains_and_full_audits(tmp_path, monkeypatch):
    monkeypatch.setattr(ui_server, "REPO_ROOT", tmp_path)
    text = "The committee approved the budget in order to begin quickly."
    data = ui_server._compress_payload({
        "mode": "tree_semantic", "text": text, "target_ratio": 0.9,
        "arm": "b0", "substitute": True,
    })
    assert data["original"] == data["tree"]["original"] == text
    assert data["pruned"] == data["tree"]["compressed"] == "The committee approved the budget in order to begin."
    assert data["semantic"]["original"] == data["pruned"]
    assert data["compressed"] == data["semantic"]["compressed"] == "The committee approved the budget to begin."
    assert (data["tokens_before"], data["tokens_after_tree"], data["tokens_after"]) == (11, 10, 8)
    assert (data["tree_saved_tokens"], data["semantic_saved_tokens"], data["saved_tokens"]) == (1, 2, 3)
    assert data["ratio"] == 8 / 11
    assert data["saved_percent"] == round(300 / 11, 1)
    assert data["tree"]["substitutions"] == {}
    assert data["tree"]["deleted"]
    assert data["semantic"]["edits"][0]["original"] == "in order to"
    assert data["semantic"]["decisions"][0]["decision"] == "selected"
    assert data["tree"]["token_counter"] == data["semantic"]["token_counter"] == data["token_counter"]
    assert data["review_status"] == "unreviewed"
    assert any("unreviewed" in warning for warning in data["warnings"])
    assert data["timings_ms"]["total"] >= data["timings_ms"]["tree"] >= 0
    assert data["timings_ms"]["total"] >= data["timings_ms"]["semantic"] >= 0


def test_tree_semantic_preserves_protected_clause_and_tree_audit(tmp_path, monkeypatch):
    monkeypatch.setattr(ui_server, "REPO_ROOT", tmp_path)
    text = "The team worked carefully, unless spending exceeds $5,000."
    data = ui_server._compress_payload({"mode": "tree_semantic", "text": text, "target_ratio": 0.6})
    protected = "unless spending exceeds $5,000"
    assert protected in data["original"]
    assert protected in data["pruned"]
    assert protected in data["compressed"]
    assert data["tree"]["vetoed"]
    assert data["tree"]["original"] == text
    assert data["semantic"]["original"] == data["pruned"]


def test_tree_semantic_no_edit_keeps_pruned_text(tmp_path, monkeypatch):
    monkeypatch.setattr(ui_server, "REPO_ROOT", tmp_path)
    text = "They must work in order to finish the assignment."
    data = ui_server._compress_payload({"mode": "tree_semantic", "text": text, "target_ratio": 1.0})
    assert data["pruned"] == data["compressed"] == text
    assert data["tokens_after_tree"] == data["tokens_after"]
    assert data["semantic_saved_tokens"] == 0
    assert data["semantic"]["edits"] == []
    assert data["semantic"]["decisions"][0]["reason"].startswith("protected_")


def test_tree_semantic_uses_semantic_provider_options(tmp_path, monkeypatch):
    monkeypatch.setattr(ui_server, "REPO_ROOT", tmp_path)
    data = ui_server._compress_payload({
        "mode": "tree_semantic", "text": "We discuss the methodologies. These projects necessitate planning.",
        "target_ratio": 1.0, "arm": "b2", "vector_source": "synthetic",
        "max_candidates": 8, "embedding_slots": 3,
        "min_embedding_similarity": 0.75, "adaptive_similarity": True,
        "encoding": "tiktoken:o200k_base", "input_price_per_million": 2.5,
    })
    assert data["semantic"]["provider_metadata"]["arm"] == "b2"
    assert data["semantic"]["provider_metadata"]["embedding_slots"] == 3
    assert "synthetic" in data["semantic"]["vector_label"].lower()
    assert data["semantic"]["input_price_per_million"] == 2.5
    assert data["semantic"]["decisions"]


@pytest.mark.parametrize("settings, message", [
    ({"target_ratio": "bad"}, "Target ratio"),
    ({"target_ratio": 0.09}, "Target ratio"),
    ({"target_ratio": 1.01}, "Target ratio"),
    ({"encoding": "cl100k_base"}, "o200k_base"),
    ({"encoding": "bad"}, "encoding"),
    ({"arm": "b3"}, "arm"),
    ({"arm": "b1", "vector_source": "local"}, "vector_path"),
])
def test_tree_semantic_rejects_invalid_inputs(settings, message, tmp_path, monkeypatch):
    monkeypatch.setattr(ui_server, "REPO_ROOT", tmp_path)
    with pytest.raises(ValueError, match=message):
        ui_server._compress_payload({
            "mode": "tree_semantic", "text": "The committee approved the budget in order to begin quickly.",
            **settings,
        })


def test_tree_semantic_discards_token_increasing_semantic_audit(tmp_path, monkeypatch):
    monkeypatch.setattr(ui_server, "REPO_ROOT", tmp_path)
    actual_semantic = ui_server._semantic_payload

    def increasing_semantic(payload, pruned, **kwargs):
        result = actual_semantic(payload, pruned, **kwargs)
        result["compressed"] = pruned + " This addition increases token use substantially."
        # The combined mode must measure the proposal, even if a provider's
        # reported count is stale or incorrect.
        result["tokens_after"] = result["tokens_before"]
        return result

    monkeypatch.setattr(ui_server, "_semantic_payload", increasing_semantic)
    data = ui_server._compress_payload({
        "mode": "tree_semantic", "text": "The committee approved the budget in order to begin quickly.",
        "target_ratio": 0.9,
    })
    assert data["compressed"] == data["pruned"] == data["semantic"]["compressed"]
    assert data["tokens_after"] == data["tokens_after_tree"] == data["semantic"]["tokens_after"]
    assert data["semantic_saved_tokens"] == data["semantic"]["saved_tokens"] == 0
    assert data["semantic"]["edits"] == data["semantic"]["occurrences"] == []
    assert not any(decision["decision"] == "selected" for decision in data["semantic"]["decisions"])
    assert any("discarded" in warning for warning in data["semantic"]["warnings"])


@pytest.mark.parametrize("text", [
    "Please review the detailed request for {customer_name} carefully.",
    'The report says "The committee approved the budget in order to begin quickly." Please review it.',
    "~~~\nThe committee approved the budget in order to begin quickly.\n~~~\nPlease review the budget carefully.",
])
def test_tree_semantic_discards_tree_deletion_inside_lexical_frozen_range(text):
    data = ui_server._compress_payload({"mode": "tree_semantic", "text": text, "target_ratio": 0.6})
    assert data["pruned"] == data["tree"]["compressed"] == text
    assert not data["tree"]["deleted"]
    assert data["tree"]["tokens_before"] == data["tree"]["tokens_after"] == data["tokens_after_tree"]
    assert data["tree_saved_tokens"] == 0
    assert any("frozen range" in warning for warning in data["tree"]["warnings"])
    assert any("frozen range" in warning for warning in data["warnings"])


def test_tree_semantic_recounts_stale_nested_stage_counts(monkeypatch):
    actual_compress = ui_server.compress
    actual_semantic = ui_server._semantic_payload
    counters = []

    def stale_tree(*args, **kwargs):
        counters.append(kwargs["counter"])
        return dataclasses.replace(actual_compress(*args, **kwargs), tokens_before=1, tokens_after=1, ratio=1.0)

    def stale_semantic(payload, pruned, **kwargs):
        counters.append(kwargs["counter"])
        result = actual_semantic(payload, pruned, **kwargs)
        result.update(tokens_before=1, tokens_after=1, saved_tokens=0, ratio=1.0, saved_percent=0)
        return result

    monkeypatch.setattr(ui_server, "compress", stale_tree)
    monkeypatch.setattr(ui_server, "_semantic_payload", stale_semantic)
    data = ui_server._compress_payload({
        "mode": "tree_semantic", "text": "The committee approved the budget in order to begin quickly.",
        "target_ratio": 0.9,
    })
    assert (data["tokens_before"], data["tokens_after_tree"], data["tokens_after"]) == (11, 10, 8)
    assert (data["tree"]["tokens_before"], data["tree"]["tokens_after"]) == (11, 10)
    assert (data["semantic"]["tokens_before"], data["semantic"]["tokens_after"]) == (10, 8)
    assert data["semantic"]["saved_tokens"] == data["semantic_saved_tokens"] == 2
    assert counters[0] is counters[1]
    assert not isinstance(counters[0], ui_server.CachedCounter)
    assert data["semantic"]["review_status"] == "unreviewed"
    assert "do not accept combined output" in data["semantic"]["workflow_disclaimer"]


@pytest.mark.parametrize("tree_output, warning", [
    ("The committee approved the budget in order to begin quickly. Extra words that increase tokens.", "increased"),
    (" ", "empty"),
])
def test_tree_semantic_discards_increasing_or_blank_tree_stage(tree_output, warning, monkeypatch):
    actual_compress = ui_server.compress
    text = "The committee approved the budget in order to begin quickly."

    def unsafe_tree(*args, **kwargs):
        return dataclasses.replace(actual_compress(*args, **kwargs), compressed=tree_output)

    monkeypatch.setattr(ui_server, "compress", unsafe_tree)
    data = ui_server._compress_payload({"mode": "tree_semantic", "text": text, "target_ratio": 0.9})
    assert data["pruned"] == data["semantic"]["original"] == text
    assert data["tree"]["compressed"] == text
    assert not data["tree"]["deleted"]
    assert data["tree_saved_tokens"] == 0
    assert data["tokens_before"] == data["tokens_after_tree"]
    assert any(warning in message for message in data["tree"]["warnings"])


def test_tree_semantic_freezes_lowercase_defined_term(monkeypatch):
    actual_compress = ui_server.compress
    text = '"service level" means the required level. The committee approved the service level quickly.'
    start = text.rindex("service level")
    end = start + len("service level")

    def deleting_defined_term(*args, **kwargs):
        result = actual_compress(*args, **kwargs)
        deletion = Deletion(Span(start, end), "service level", "obj", 0.0, "test")
        return dataclasses.replace(
            result, compressed=text[:start] + text[end:], deleted=(deletion,),
        )

    monkeypatch.setattr(ui_server, "compress", deleting_defined_term)
    data = ui_server._compress_payload({"mode": "tree_semantic", "text": text})
    assert data["pruned"] == text
    assert not data["tree"]["deleted"]
    assert data["tree_saved_tokens"] == 0
    assert any("frozen range" in warning for warning in data["tree"]["warnings"])


def test_tree_semantic_discards_zero_saving_rewrite(monkeypatch):
    actual_semantic = ui_server._semantic_payload

    def zero_saving_semantic(payload, pruned, **kwargs):
        result = actual_semantic(payload, pruned, **kwargs)
        result["compressed"] = pruned.replace("The committee", "A committee", 1)
        result["tokens_after"] = result["tokens_before"] - 2
        return result

    monkeypatch.setattr(ui_server, "_semantic_payload", zero_saving_semantic)
    data = ui_server._compress_payload({
        "mode": "tree_semantic", "text": "The committee approved the budget in order to begin quickly.",
        "target_ratio": 0.9,
    })
    assert data["compressed"] == data["pruned"]
    assert data["semantic_saved_tokens"] == data["semantic"]["saved_tokens"] == 0
    assert data["semantic"]["tokens_before"] == data["semantic"]["tokens_after"]
    assert data["semantic"]["edits"] == data["semantic"]["occurrences"] == []
    assert not any(item["decision"] == "selected" for item in data["semantic"]["decisions"])
    assert any(item["reason"] == "combined_stage_no_savings" for item in data["semantic"]["decisions"])


@pytest.mark.parametrize("payload, message", [
    ({"mode": "tree_semantic", "target_ratio": True}, "Target ratio"),
    ({"mode": "tree_semantic", "encoding": "cl100k_base"}, "o200k_base"),
    ({"mode": "tree_semantic", "arm": "bad"}, "arm"),
    ({"mode": "tree_semantic", "input_price_per_million": -1}, "input_price_per_million"),
    ({"mode": "tree_semantic", "protected_spans": []}, "protected_spans"),
    ({"mode": "semantic", "protected_spans": []}, "protected_spans"),
])
def test_semantic_invalid_inputs_fail_before_tree(payload, message, monkeypatch):
    def unexpected_tree(*args, **kwargs):
        pytest.fail("tree compression ran before input validation")

    monkeypatch.setattr(ui_server, "compress", unexpected_tree)
    with pytest.raises(ValueError, match=message):
        ui_server._compress_payload({
            "text": "The committee approved the budget in order to begin quickly.", **payload,
        })


@pytest.mark.parametrize("handler", [
    "_semantic_preview_payload", "_semantic_approve_payload", "_semantic_validate_payload",
])
@pytest.mark.parametrize("field", [{"mode": "tree_semantic"}, {"target_ratio": 0.9}])
def test_semantic_approval_endpoints_reject_combined_fields(handler, field):
    with pytest.raises(ValueError, match="tree_semantic"):
        getattr(ui_server, handler)({"text": "A prompt.", **field})


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
    assert first["vector_size"] == {"source_rows": 12, "retained_rows": 12, "dimensions": 4}
    path.write_text(source.replace("12 4\n", "13 4\n", 1) + "extra 0 0 0 1\n", encoding="utf-8")
    changed = ui_server._compress_payload(payload)
    assert ui_server._vector_index.cache_info().misses == 2
    assert changed["provider_metadata"]["vectors"]["source_rows"] == 13


def test_semantic_local_vector_retention_reaches_index_and_response(tmp_path, monkeypatch):
    path = tmp_path / "vectors.txt"
    path.write_text(ui_server.SYNTHETIC_VECTORS.read_text(encoding="utf-8"), encoding="utf-8")
    requested = []
    def capture_index(source, modified_ns, size, max_words):
        requested.append(max_words)
        return ui_server.VectorIndex.from_text(source, max_words=max_words)
    monkeypatch.setattr(ui_server, "_vector_index", capture_index)
    result = ui_server._compress_payload({
        "mode": "semantic", "text": "Write a short answer.", "arm": "b1",
        "vector_source": "local", "vector_path": str(path), "vector_max_words": 400000,
    })
    assert requested == [400000]
    assert result["vector_size"] == {"source_rows": 12, "retained_rows": 12, "dimensions": 4}
    assert result["provider_metadata"]["vectors"]["max_words"] == 400000


@pytest.mark.parametrize("settings, message", [
    ({"arm": "b3"}, "arm"),
    ({"arm": ["b0"]}, "arm"),
    ({"use_wordnet": "false"}, "use_wordnet"),
    ({"vector_source": "remote"}, "vector_source"),
    ({"vector_source": ["synthetic"]}, "vector_source"),
    ({"vector_max_words": True}, "vector_max_words"),
    ({"vector_max_words": "400000"}, "vector_max_words"),
    ({"vector_max_words": 250000}, "vector_max_words"),
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
            "mode": "tree_semantic", "arm": "b0", "target_ratio": 0.9,
            "text": "The committee approved the budget in order to begin quickly.",
        }), headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        combined = json.loads(response.read())
        assert response.status == 200
        assert combined["mode"] == "tree_semantic"
        assert (combined["tokens_before"], combined["tokens_after_tree"], combined["tokens_after"]) == (11, 10, 8)
        assert combined["tree"]["original"] == combined["original"]
        assert combined["semantic"]["original"] == combined["pruned"]

        connection.request("POST", "/api/compress", body=json.dumps({
            "mode": "tree_semantic", "encoding": "cl100k_base",
            "text": "The committee approved the budget in order to begin quickly.",
        }), headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        error = json.loads(response.read())
        assert response.status == 400
        assert "o200k_base" in error["error"]

        for endpoint in ("preview", "approve", "validate"):
            connection.request("POST", f"/api/semantic/{endpoint}", body=json.dumps({
                "mode": "tree_semantic", "text": "A prompt.",
            }), headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            error = json.loads(response.read())
            assert response.status == 400
            assert "tree_semantic" in error["error"]

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


def test_semantic_vector_retention_selectable_and_metadata():
    text = "We discuss the methodologies. These projects necessitate planning."
    # Default 50k
    default_res = ui_server._compress_payload({
        "mode": "semantic", "text": text, "arm": "b2", "vector_source": "synthetic",
    })
    assert default_res["provider_metadata"]["vectors"]["max_words"] == 50_000
    assert default_res["vector_size"] == {"source_rows": 12, "retained_rows": 12, "dimensions": 4}
    assert not any("400k" in w.lower() for w in default_res["warnings"])
    assert "synthetic" in default_res["vector_label"].lower()

    # Selectable 10k
    res_10k = ui_server._compress_payload({
        "mode": "semantic", "text": text, "arm": "b2", "vector_source": "synthetic",
        "vector_max_words": 10_000,
    })
    assert res_10k["provider_metadata"]["vectors"]["max_words"] == 10_000
    assert res_10k["vector_size"] == {"source_rows": 12, "retained_rows": 12, "dimensions": 4}
    assert not any("400k" in w.lower() for w in res_10k["warnings"])

    # Selectable 400k with cold load warning
    res_400k = ui_server._compress_payload({
        "mode": "semantic", "text": text, "arm": "b2", "vector_source": "synthetic",
        "vector_max_words": 400_000,
    })
    assert res_400k["provider_metadata"]["vectors"]["max_words"] == 400_000
    assert res_400k["vector_size"] == {"source_rows": 12, "retained_rows": 12, "dimensions": 4}
    assert not any("400k" in w.lower() for w in res_400k["warnings"])


def test_semantic_retention_affects_index_cache_and_config_hash(tmp_path):
    source = ui_server.SYNTHETIC_VECTORS.read_text(encoding="utf-8")
    path = tmp_path / "vectors.txt"
    path.write_text(source, encoding="utf-8")
    ui_server._vector_index.cache_clear()

    res_10k = ui_server._compress_payload({
        "mode": "semantic", "text": "Write a short answer.", "arm": "b1",
        "vector_source": "local", "vector_path": str(path), "vector_max_words": 10_000,
    })
    res_50k = ui_server._compress_payload({
        "mode": "semantic", "text": "Write a short answer.", "arm": "b1",
        "vector_source": "local", "vector_path": str(path), "vector_max_words": 50_000,
    })
    # Different retentions must produce distinct config hashes
    assert res_10k["config_hash"] != res_50k["config_hash"]
    # And separate cache misses
    assert ui_server._vector_index.cache_info().misses == 2


def test_semantic_approval_and_validation_enforce_retention_setting(tmp_path, monkeypatch):
    monkeypatch.setattr(ui_server, "_TEMPLATE_STORE", ui_server.TemplateStore(tmp_path / "templates"))
    text = "They worked in order to finish the assignment."
    comp_50k = ui_server._compress_payload({
        "mode": "semantic", "text": text, "arm": "b1", "vector_source": "synthetic",
        "vector_max_words": 50_000,
    })
    decisions = [
        {"span": {"start": e["span"]["start"], "end": e["span"]["end"]},
         "original": e["original"], "replacement": e["replacement"], "decision": "accepted"}
        for e in comp_50k["edits"]
    ]

    # Approving with mismatching retention setting fails config_hash verification
    with pytest.raises(ValueError, match="Configuration hash mismatch"):
        ui_server._semantic_approve_payload({
            "text": text, "arm": "b1", "vector_source": "synthetic",
            "vector_max_words": 10_000,
            "config_hash": comp_50k["config_hash"],
            "vector_identity": comp_50k["vector_label"],
            "decisions": decisions,
        })

    # Approving with same retention setting succeeds
    approved = ui_server._semantic_approve_payload({
        "text": text, "arm": "b1", "vector_source": "synthetic",
        "vector_max_words": 50_000,
        "config_hash": comp_50k["config_hash"],
        "vector_identity": comp_50k["vector_label"],
        "decisions": decisions,
    })
    template_id = approved["template"]["template_id"]

    # Validating with same retention setting passes
    valid_res = ui_server._semantic_validate_payload({
        "text": text, "template_id": template_id,
        "arm": "b1", "vector_source": "synthetic", "vector_max_words": 50_000,
    })
    assert valid_res["is_valid"] is True

    # Validating with different retention setting (stale config) invalidates the template
    invalid_res = ui_server._semantic_validate_payload({
        "text": text, "template_id": template_id,
        "arm": "b1", "vector_source": "synthetic", "vector_max_words": 10_000,
    })
    assert invalid_res["is_valid"] is False
    assert invalid_res["status"] == "invalidated"
    assert "Configuration hash mismatch" in invalid_res["invalidation_reason"]

    # Validating with invalid retention value is rejected
    with pytest.raises(ValueError, match="vector_max_words"):
        ui_server._semantic_validate_payload({
            "text": text, "template_id": template_id,
            "arm": "b1", "vector_source": "synthetic", "vector_max_words": 25000,
        })
