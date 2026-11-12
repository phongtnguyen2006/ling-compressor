"""Paired AG News proxy evidence integrity and metrics with a fake classifier."""
import hashlib
import json

import pytest

from eval import ag_news_accuracy as accuracy


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class FakeVectorizer:
    def fit_transform(self, texts):
        self.vocabulary_ = {word: index for index, word in enumerate(sorted(
            {word for text in texts for word in text.split()}))}
        return texts

    def transform(self, texts):
        return texts

    def get_params(self):
        return {"kind": "fixture"}


class FakeClassifier:
    def fit(self, texts, labels):
        self.classes_ = sorted(set(labels))
        self.coef_ = [[1, 0], [0, 1]]
        self.intercept_ = [0, 0]
        return self

    def predict(self, texts):
        return [1 if "sports" in text else 0 for text in texts]

    def get_params(self):
        return {"kind": "fixture"}


def _fixture(tmp_path):
    parquet = pytest.importorskip("pyarrow.parquet")
    pa = pytest.importorskip("pyarrow")
    pilot = tmp_path / "pilot"
    pilot.mkdir()
    docs = [
        {"id": "dev", "text": "development world", "group": "dev", "split": "development",
         "gold_label": 0},
        {"id": "a", "text": "world report", "group": "a", "split": "held-out", "gold_label": 0},
        {"id": "b", "text": "sports report", "group": "b", "split": "held-out", "gold_label": 1},
    ]
    for doc in docs:
        doc["source_sha256"] = _sha(doc["text"])
    corpus = tmp_path / "corpus.jsonl"
    _write_jsonl(corpus, docs)
    inputs = [{key: doc[key] for key in ("id", "text", "group", "split")} for doc in docs]
    _write_jsonl(pilot / "inputs.jsonl", inputs)
    identity = {"schema_version": "lexical-pilot-v1",
                "inputs": [{"id": doc["id"], "group": doc["group"], "split": doc["split"],
                            "sha256": doc["source_sha256"]} for doc in docs],
                "counter": "fixture", "providers": {arm: {"name": arm}
                                                    for arm in ("u", *accuracy.ARMS)},
                "configuration": {}, "versions": {}, "software_sha256": {}}
    manifest = {**identity, "run_fingerprint": accuracy._digest(identity), "corpus_label": "fixture"}
    (pilot / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    rows = []
    for doc in docs:
        for arm in ("u", *accuracy.ARMS):
            suggested = doc["text"]
            if doc["id"] == "a" and arm == "b1":
                suggested = "sports report"  # Correct original becomes wrong.
            rows.append({"doc_id": doc["id"], "arm": arm, "original": doc["text"],
                         "suggested": suggested, "group": doc["group"], "split": doc["split"]})
    _write_jsonl(pilot / "results.jsonl", rows)
    _checksums(pilot)
    train = tmp_path / "train.parquet"
    train_rows = [{"text": "development world", "label": 0}]
    train_rows += [{"text": f"training {label} unique", "label": label} for label in range(4)]
    parquet.write_table(pa.Table.from_pylist(train_rows), train)
    return pilot, train, corpus, docs


def _checksums(pilot):
    checksums = {name: accuracy._file_sha(pilot / name)
                 for name in ("manifest.json", "inputs.jsonl", "results.jsonl")}
    (pilot / "checksums.json").write_text(json.dumps(checksums), encoding="utf-8")


def _evaluate(fixture, out):
    pilot, train, corpus, _ = fixture
    return accuracy.evaluate(pilot, train, corpus, out, train_per_label=1,
                             vectorizer_factory=FakeVectorizer,
                             classifier_factory=FakeClassifier)


def test_paired_accuracy_drop_changed_subset_raw_predictions_and_identity(tmp_path):
    fixture = _fixture(tmp_path)
    first = _evaluate(fixture, tmp_path / "first")
    second = _evaluate(fixture, tmp_path / "second")
    b1 = first["summary"]["b1"]
    assert b1 == {"documents": 2, "original_accuracy": 1.0,
                  "suggested_accuracy": 0.5, "paired_accuracy_delta": -0.5,
                  "changed_only": {"documents": 1, "original_accuracy": 1.0,
                                   "suggested_accuracy": 0.0, "paired_accuracy_delta": -1.0}}
    assert first["summary"]["b0"]["changed_only"]["paired_accuracy_delta"] is None
    assert first["predictions"][0]["arms"]["b1"]["prediction"] == 1
    assert first["predictions"][0]["original_prediction"] == 0
    assert first["identity"] == second["identity"]
    assert (tmp_path / "first" / "report.json").read_bytes() == (tmp_path / "second" / "report.json").read_bytes()
    assert fixture[3][0]["source_sha256"] not in first["identity"]["training_text_sha256"]
    assert "not LLM downstream accuracy" in first["scope"]
    assert first["source_license"].startswith("unknown")
    assert first["production_approval"] is False
    assert "Changed-only delta" in (tmp_path / "first" / "report.md").read_text()
    with pytest.raises(ValueError, match="already exists"):
        _evaluate(fixture, tmp_path / "first")


def test_rejects_missing_or_duplicate_pair_even_with_valid_checksums(tmp_path):
    fixture = _fixture(tmp_path)
    pilot = fixture[0]
    rows = accuracy._jsonl(pilot / "results.jsonl")
    _write_jsonl(pilot / "results.jsonl", rows[:-1])
    _checksums(pilot)
    with pytest.raises(ValueError, match="pairs incomplete"):
        _evaluate(fixture, tmp_path / "missing")
    _write_jsonl(pilot / "results.jsonl", rows + [rows[-1]])
    _checksums(pilot)
    with pytest.raises(ValueError, match="duplicate pilot pair"):
        _evaluate(fixture, tmp_path / "duplicate")


def test_rejects_tampered_hash_original_and_gold_label(tmp_path):
    fixture = _fixture(tmp_path)
    pilot, _, corpus, docs = fixture
    rows = accuracy._jsonl(pilot / "results.jsonl")
    rows[0]["suggested"] = "tampered"
    _write_jsonl(pilot / "results.jsonl", rows)
    with pytest.raises(ValueError, match="checksum mismatch"):
        _evaluate(fixture, tmp_path / "checksum")
    rows[0]["original"] = "tampered"
    _write_jsonl(pilot / "results.jsonl", rows)
    _checksums(pilot)
    with pytest.raises(ValueError, match="pair/original mismatch"):
        _evaluate(fixture, tmp_path / "original")
    rows[0]["original"] = docs[0]["text"]
    rows[0]["suggested"] = docs[0]["text"]
    _write_jsonl(pilot / "results.jsonl", rows)
    _checksums(pilot)
    changed = [dict(doc) for doc in docs]
    changed[0]["gold_label"] = 1
    _write_jsonl(corpus, changed)
    with pytest.raises(ValueError, match="gold label mismatches"):
        _evaluate(fixture, tmp_path / "label")


def test_rejects_selected_cross_split_duplicate_and_fingerprint_change(tmp_path):
    fixture = _fixture(tmp_path)
    pilot, _, corpus, docs = fixture
    duplicate = [dict(doc) for doc in docs]
    duplicate[1]["text"] = duplicate[0]["text"]
    duplicate[1]["source_sha256"] = duplicate[0]["source_sha256"]
    _write_jsonl(corpus, duplicate)
    inputs = accuracy._jsonl(pilot / "inputs.jsonl")
    inputs[1]["text"] = duplicate[0]["text"]
    _write_jsonl(pilot / "inputs.jsonl", inputs)
    rows = accuracy._jsonl(pilot / "results.jsonl")
    for row in rows:
        if row["doc_id"] == "a":
            row["original"] = duplicate[0]["text"]
            row["suggested"] = duplicate[0]["text"]
    _write_jsonl(pilot / "results.jsonl", rows)
    manifest = json.loads((pilot / "manifest.json").read_text())
    manifest["inputs"][1]["sha256"] = duplicate[0]["source_sha256"]
    manifest["run_fingerprint"] = accuracy._digest(
        {key: manifest[key] for key in accuracy.PILOT_IDENTITY_KEYS})
    (pilot / "manifest.json").write_text(json.dumps(manifest))
    _checksums(pilot)
    with pytest.raises(ValueError, match="duplicate selected original hash"):
        _evaluate(fixture, tmp_path / "duplicate_hash")
    _write_jsonl(corpus, docs)
    manifest["configuration"] = {"changed": True}
    (pilot / "manifest.json").write_text(json.dumps(manifest))
    _checksums(pilot)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        _evaluate(fixture, tmp_path / "fingerprint")


def test_corpus_superset_is_excluded_from_training_and_real_model_fits(tmp_path):
    fixture = _fixture(tmp_path)
    pilot, train, corpus, docs = fixture
    extra = {"id": "unused", "text": "extra world article", "group": "unused",
             "split": "development", "gold_label": 0,
             "source_sha256": _sha("extra world article")}
    _write_jsonl(corpus, docs + [extra])
    pa = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    rows = parquet.read_table(train).to_pylist()
    rows.append({"text": extra["text"], "label": 0})
    parquet.write_table(pa.Table.from_pylist(rows), train)
    report = accuracy.evaluate(pilot, train, corpus, tmp_path / "real", train_per_label=1)
    assert report["held_out_documents"] == 2
    assert len(report["identity"]["training_text_sha256"]) == 4
    assert extra["source_sha256"] not in report["identity"]["training_text_sha256"]
    assert report["identity"]["classifier"]["class"].endswith("LinearSVC")
    assert report["identity"]["vectorizer"]["fitted_state_sha256"]
    second = accuracy.evaluate(pilot, train, corpus, tmp_path / "real_again", train_per_label=1)
    assert second == report


def test_ambiguous_unselected_train_hash_is_excluded_and_counted(tmp_path):
    fixture = _fixture(tmp_path)
    pilot, train, corpus, docs = fixture
    pa = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    ambiguous_text = "same article with two official labels"
    rows = parquet.read_table(train).to_pylist()
    rows.extend([{"text": ambiguous_text, "label": 0},
                 {"text": ambiguous_text, "label": 1},
                 {"text": ambiguous_text, "label": 0},
                 {"text": docs[1]["text"], "label": docs[1]["gold_label"]}])
    parquet.write_table(pa.Table.from_pylist(rows), train)

    report = _evaluate(fixture, tmp_path / "ambiguous")
    hashes = report["identity"]["training_text_sha256"]
    assert report["identity"]["ambiguous_train_hashes_excluded"] == 1
    assert _sha(ambiguous_text) not in hashes
    assert docs[0]["source_sha256"] not in hashes  # Development leakage guard.
    assert docs[1]["source_sha256"] not in hashes  # Held-out leakage guard.
    assert "Ambiguous-label training text hashes excluded: 1" in (
        tmp_path / "ambiguous" / "report.md").read_text()


def test_selected_hash_with_conflicting_official_labels_is_rejected(tmp_path):
    fixture = _fixture(tmp_path)
    _, train, _, docs = fixture
    pa = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    rows = parquet.read_table(train).to_pylist()
    rows.append({"text": docs[0]["text"], "label": 1})
    parquet.write_table(pa.Table.from_pylist(rows), train)
    with pytest.raises(ValueError, match="selected gold label mismatches"):
        _evaluate(fixture, tmp_path / "conflicting_selected")
