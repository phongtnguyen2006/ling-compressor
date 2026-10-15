from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import subprocess
import sys

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


@pytest.mark.parametrize("payload", [{}, {"text": "  "}, {"text": "hello", "target_ratio": 2}, {"text": "hello", "mode": "unknown"}])
def test_compress_payload_rejects_invalid_input(payload):
    with pytest.raises(ValueError):
        ui_server._compress_payload(payload)
