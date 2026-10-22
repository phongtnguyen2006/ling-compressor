"""Evidence integrity, unbiased denominators, and offline CLI controls."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re

import pytest

from eval import lexical_pilot as pilot
from promptcomp.lexical import PilotProvider


class WordCounter:
    name = "test:words-and-punctuation"

    def count(self, text):
        return len(re.findall(r"\w+|[^\w\s]", text))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def example_docs():
    return [pilot.PilotDocument("a", "They worked in order to finish.", "family-a"),
            pilot.PilotDocument("b", "Write a short answer.", "family-b")]


def test_pilot_exports_actual_occurrences_blind_reviews_and_all_input_denominator(tmp_path):
    counter = WordCounter()
    provider = PilotProvider(phrases={"in order to": "to"})
    result = pilot.run_pilot(example_docs(), {"u": None, "b0": provider}, counter,
                             tmp_path, corpus_label="authored test fixtures")
    rows = read_jsonl(tmp_path / "results.jsonl")
    assert len(rows) == 4
    summary = next(row for row in result["summary"] if row["arm"] == "b0")
    assert summary["tokens_before"] == sum(counter.count(doc.text) for doc in example_docs())
    assert summary["tokens_saved"] == 2
    assert summary["changed_documents"] == 1
    assert summary["corpus_reduction"] == pytest.approx(2 / summary["tokens_before"])
    assert summary["semantic_quality"] == "not_evaluated"
    for row in rows:
        for edit in row["edits"]:
            assert row["original"][edit["span"]["start"]:edit["span"]["end"]] == edit["original"]
    edit_reviews = read_jsonl(tmp_path / "edit-review.jsonl")
    assert edit_reviews
    for review in edit_reviews + read_jsonl(tmp_path / "prompt-review.jsonl"):
        assert review["meaning_preserved"] is None
        assert review["reviewer"] is None
        assert "provenance" not in review and "arm" not in review
    assert read_jsonl(tmp_path / "review-key.jsonl")
    pools = read_jsonl(tmp_path / "candidate-pools.jsonl")
    assert pools and any(pool["source"] == "in order to" for pool in pools)
    assert all(pool["baseline_pool"] for pool in pools)
    assert "curated rules only" in (tmp_path / "summary.md").read_text()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["providers"]["b0"]["curated"]["sha256"]
    assert manifest["software_sha256"]
    assert read_jsonl(tmp_path / "inputs.jsonl") == [asdict(doc) for doc in example_docs()]
    for name, digest in json.loads((tmp_path / "checksums.json").read_text()).items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == digest


def test_repeated_runs_have_same_identity_and_blank_judgments(tmp_path):
    kwargs = {"documents": example_docs(), "providers": {"b0": PilotProvider()},
              "counter": WordCounter(), "corpus_label": "fixture"}
    first = pilot.run_pilot(output_dir=tmp_path / "first", **kwargs)
    second = pilot.run_pilot(output_dir=tmp_path / "second", **kwargs)
    assert first["run_fingerprint"] == second["run_fingerprint"]
    assert (tmp_path / "first" / "edit-review.jsonl").read_bytes() == (tmp_path / "second" / "edit-review.jsonl").read_bytes()
    changed = pilot.run_pilot([pilot.PilotDocument("a", "Different text.", "family-a")],
                              kwargs["providers"], kwargs["counter"], tmp_path / "third", corpus_label="fixture")
    assert first["run_fingerprint"] != changed["run_fingerprint"]


def test_existing_evidence_is_never_overwritten(tmp_path):
    evidence = tmp_path / "manifest.json"
    evidence.write_text("old evidence")
    with pytest.raises(ValueError, match="not overwritten"):
        pilot.run_pilot(example_docs(), {"u": None}, WordCounter(), tmp_path, corpus_label="fixture")
    assert evidence.read_text() == "old evidence"


def test_empty_input_is_accounted_for_without_division_error(tmp_path):
    result = pilot.run_pilot([pilot.PilotDocument("empty", "", "empty")], {"u": None},
                             WordCounter(), tmp_path, corpus_label="fixture")
    row = result["summary"][0]
    assert row["documents"] == 1
    assert row["tokens_before"] == 0
    assert row["corpus_reduction"] == 0


@pytest.mark.parametrize("contents", [
    '[{"id":"a","text":"x"}]\n',
    '{"id":"a","text":42}\n',
    '{"id":"","text":"x"}\n',
    '{"id":"a"}\n',
    '{"id":"a","text":"x"}\n{"id":"a","text":"y"}\n',
])
def test_invalid_manifest_is_rejected_before_candidate_discovery(tmp_path, contents):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(contents)
    with pytest.raises(ValueError):
        pilot.load_documents(corpus)


def test_manifest_preserves_preselected_group_and_split(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('{"id":"x","text":"Text.","group":"template-7","split":"holdout"}\n')
    assert pilot.load_documents(corpus) == [pilot.PilotDocument("x", "Text.", "template-7", "holdout")]


def test_cli_missing_vectors_is_explicit_not_a_silent_baseline(tmp_path, capsys):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("Text.")
    assert pilot.main(["--corpus", str(corpus), "--out", str(tmp_path / "out"), "--arms", "b1"]) == 2
    assert "require --vectors" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_cli_selects_documents_before_discovery_reproducibly(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for number in range(10):
        (corpus / f"d{number}.txt").write_text(f"Example {number}.")
    monkeypatch.setattr(pilot, "_local_counter", lambda _: WordCounter())
    for name in ("first", "second"):
        assert pilot.main(["--corpus", str(corpus), "--out", str(tmp_path / name),
                           "--arms", "u", "--max-documents", "3", "--seed", "7"]) == 0
    assert (tmp_path / "first" / "inputs.jsonl").read_bytes() == (tmp_path / "second" / "inputs.jsonl").read_bytes()
    assert len(read_jsonl(tmp_path / "first" / "inputs.jsonl")) == 3


def test_tokenizer_cache_miss_cannot_trigger_hidden_network_download(monkeypatch):
    import tiktoken
    import tiktoken.load

    original_reader = tiktoken.load.read_file
    monkeypatch.setattr(tiktoken, "get_encoding", lambda _: tiktoken.load.read_file("https://example.test/encoding"))
    with pytest.raises(RuntimeError, match="Explicitly initialize"):
        pilot._local_counter("o200k_base")
    assert tiktoken.load.read_file is original_reader


def test_synthetic_vector_warning_is_exported_with_measurements(tmp_path):
    result = pilot.run_pilot(example_docs(), {"u": None}, WordCounter(), tmp_path,
                             corpus_label="synthetic integration fixture",
                             configuration={"synthetic_vectors": True})
    assert result["summary"][0]["tokens_saved"] == 0
    assert "SYNTHETIC VECTOR FIXTURE" in (tmp_path / "summary.md").read_text()
