"""Stored-evidence reconciliation and explicit offline embedding measurement."""
import json
from pathlib import Path
import shutil
import sys
import types

import pytest

from eval import semantic_benchmark as benchmark


EVIDENCE = (Path(__file__).resolve().parents[1] / "docs" / "semantic-token-compression"
            / "evidence" / "2026-10-07-lexical-smoke")


def test_bundled_evidence_offline_cli_has_no_semantic_claim(tmp_path, monkeypatch):
    # An installed optional package must never be imported by the default path.
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    output = tmp_path / "report"
    assert benchmark.main(["--run", str(EVIDENCE), "--out", str(output)]) == 0
    report = json.loads((output / "report.json").read_text())
    assert report["schema_version"] == "semantic-benchmark-v1"
    assert [row["arm"] for row in report["arms"]] == ["u", "b0", "b1", "b2", "b2_union"]
    baseline, changed = report["arms"][:2]
    assert (baseline["tokens_before"], baseline["tokens_after"], baseline["tokens_saved"]) == (68, 68, 0)
    assert (changed["tokens_before"], changed["tokens_after"], changed["tokens_saved"]) == (68, 62, 6)
    assert changed["token_reduction"] == pytest.approx(6 / 68)
    assert changed["coverage"] == {"result_pairs": 6, "expected_pairs": 6,
                                   "changed_outputs": 2, "unchanged_outputs": 4}
    assert changed["semantic_preservation"] == {
        "status": "not_computed", "metric": None, "mean_cosine_changed_outputs": None,
        "evaluated_pairs": 0, "eligible_changed_pairs": 2,
    }
    assert "not_computed" in (output / "summary.md").read_text()
    assert "not proof" in (output / "summary.md").read_text()


def test_local_model_is_explicit_and_scores_only_changed_pairs(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    calls = []

    class FakeModel:
        def __init__(self, path, *, local_files_only):
            calls.append((path, local_files_only))

        def encode(self, texts, **kwargs):
            assert kwargs["normalize_embeddings"] is True
            assert kwargs["show_progress_bar"] is False
            return [[1.0, 0.0] if "in order to" in text else [0.8, 0.6]
                    for text in texts]

    monkeypatch.setitem(sys.modules, "sentence_transformers",
                        types.SimpleNamespace(SentenceTransformer=FakeModel))
    output = tmp_path / "report"
    assert benchmark.main(["--run", str(EVIDENCE), "--out", str(output),
                           "--model-path", str(model_dir)]) == 0
    report = json.loads((output / "report.json").read_text())
    assert calls == [(str(model_dir.resolve()), True)]
    assert report["embedding_model_path"] == str(model_dir.resolve())
    assert report["arms"][0]["semantic_preservation"]["status"] == "no_changed_outputs"
    metric = report["arms"][1]["semantic_preservation"]
    assert metric["metric"] == "sentence_transformer_cosine"
    assert metric["evaluated_pairs"] == metric["eligible_changed_pairs"] == 2
    assert -1 <= metric["mean_cosine_changed_outputs"] <= 1
    assert "cosine" in (output / "summary.md").read_text().lower()


def test_missing_local_model_fails_without_output(tmp_path):
    output = tmp_path / "report"
    assert benchmark.main(["--run", str(EVIDENCE), "--out", str(output),
                           "--model-path", str(tmp_path / "missing")]) == 2
    assert not output.exists()


def test_bad_source_checksum_is_rejected(tmp_path):
    run = tmp_path / "run"
    shutil.copytree(EVIDENCE, run)
    with (run / "results.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        benchmark.benchmark(run)


@pytest.mark.parametrize("mutation, message", [
    (lambda rows: rows.pop(), "cover every input"),
    (lambda rows: rows.append(rows[0]), "cover every input"),
    (lambda rows: rows[0].update(tokens_after=999), "token count increased"),
    (lambda rows: rows[0].update(original="tampered"), "original text hash mismatch"),
])
def test_rejects_inconsistent_results_even_without_checksums(tmp_path, mutation, message):
    run = tmp_path / "run"
    run.mkdir()
    for name in ("manifest.json", "summary.json", "results.jsonl"):
        shutil.copyfile(EVIDENCE / name, run / name)
    rows = [json.loads(line) for line in (run / "results.jsonl").read_text().splitlines()]
    mutation(rows)
    (run / "results.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError, match=message):
        benchmark.benchmark(run)


def test_rejects_summary_mismatch_and_refuses_to_overwrite(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    for name in ("manifest.json", "summary.json", "results.jsonl"):
        shutil.copyfile(EVIDENCE / name, run / name)
    summary = json.loads((run / "summary.json").read_text())
    summary[1]["tokens_saved"] = 100
    (run / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="summary mismatch"):
        benchmark.benchmark(run)
    output = tmp_path / "existing"
    output.mkdir()
    (output / "keep.txt").write_text("keep")
    assert benchmark.main(["--run", str(EVIDENCE), "--out", str(output)]) == 2
    assert (output / "keep.txt").read_text() == "keep"


def test_rejects_manifest_identity_mismatch(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    for name in ("manifest.json", "summary.json", "results.jsonl"):
        shutil.copyfile(EVIDENCE / name, run / name)
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["counter"] = "unknown"
    (run / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        benchmark.benchmark(run)
