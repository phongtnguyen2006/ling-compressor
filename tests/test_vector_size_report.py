"""Frozen vector-size sweep reporting uses only validated pilot evidence."""
import hashlib
import json

import pytest

from eval import lexical_pilot as pilot
from eval import vector_size_report as sizes


DOCS = [("dev", "Development text.", "development"),
        ("held", "Held out text.", "held-out")]
ARMS = ["u", "b0", "b1", "b2"]
SAVINGS = {"10k": (0, 1, 2, 3), "50k": (0, 1, 3, 4), "400k": (0, 1, 2, 5)}


def _write_run(path, label, *, docs=DOCS, config_change=None, vector_change=None,
               measurements=None, input_tokens=10):
    path.mkdir()
    limit = sizes.SIZES[label]
    vector = {"sha256": "a" * 64, "format": "glove-text", "dimensions": 50,
              "source_rows": 400_000, "retained_rows": limit, "max_words": limit,
              "source_path": f"/local/{label}.txt", "name": f"vectors:{label}",
              "dtype": "float32"}
    if vector_change:
        vector.update(vector_change)
    inputs = [{"id": doc_id, "group": doc_id, "split": split,
               "sha256": hashlib.sha256(text.encode()).hexdigest()} for doc_id, text, split in docs]
    configuration = {"arms": ARMS, "max_words": limit, "wordnet": True,
                     "encoding": "o200k_base", "max_candidates": 8,
                     "embedding_slots": 4, "vector_label": f"first {label} vectors"}
    if config_change:
        configuration.update(config_change)
    providers = {arm: ({"name": "unchanged"} if arm == "u" else
                       {"name": f"pilot:{arm}:{label}", "arm": arm, "max_candidates": 8,
                        "vectors": vector}) for arm in ARMS}
    identity = {"schema_version": "lexical-pilot-v1", "inputs": inputs,
                "counter": "test-counter", "providers": providers,
                "configuration": configuration, "versions": {}, "software_sha256": {}}
    manifest = {**identity, "run_fingerprint": pilot._digest(identity), "corpus_label": "fixture"}
    if measurements is not None:
        manifest["measurements"] = measurements
    rows = []
    active_arms = configuration["arms"]
    for doc_id, original, split in docs:
        for arm in active_arms:
            saving = SAVINGS[label][ARMS.index(arm)]
            rows.append({"doc_id": doc_id, "group": doc_id, "split": split, "arm": arm,
                         "original": original, "suggested": original if not saving else f"{original} edited",
                         "tokens_before": input_tokens, "tokens_after": input_tokens - saving,
                         "token_counter": "test-counter", "edits": [{}] if saving else [],
                         "timings_ms": {"total": {"u": 0, "b0": 1, "b1": 2, "b2": 3}[arm]}})
    summary = [{"arm": arm, "documents": len(docs), "tokens_before": len(docs) * input_tokens,
                "tokens_after": sum(row["tokens_after"] for row in rows if row["arm"] == arm),
                "tokens_saved": sum(row["tokens_before"] - row["tokens_after"]
                                    for row in rows if row["arm"] == arm),
                "changed_documents": sum(row["original"] != row["suggested"]
                                         for row in rows if row["arm"] == arm)} for arm in active_arms]
    (path / "manifest.json").write_text(json.dumps(manifest))
    (path / "results.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    (path / "summary.json").write_text(json.dumps(summary))
    (path / "checksums.json").write_text(json.dumps({name: hashlib.sha256((path / name).read_bytes()).hexdigest()
                                                       for name in ("manifest.json", "results.jsonl", "summary.json")}))
    return path


def _runs(tmp_path, **changes):
    return {label: _write_run(tmp_path / label, label, **changes.get(label, {})) for label in sizes.SIZES}


def test_report_coverage_savings_runtime_and_memory(tmp_path):
    runs = _runs(tmp_path, **{"50k": {"measurements": {"peak_rss_bytes": 123_456}}})
    report = sizes.build_report(runs)
    assert report["corpus_documents"] == 2
    assert report["splits"] == {"development": 1, "held-out": 1}
    assert report["quality"] == "not_evaluated"
    assert [run["retained_rows"] for run in report["runs"]] == [10_000, 50_000, 400_000]
    assert all(run["source_rows"] == 400_000 for run in report["runs"])
    first = report["runs"][0]["splits"]["all"]
    assert first["arms"]["b2"]["all_input_tokens"] == 20
    assert first["arms"]["b2"]["tokens_saved"] == 6
    assert first["arms"]["b2"]["all_input_reduction"] == 0.3
    assert first["arms"]["b2"]["changed_documents"] == 2
    assert first["arms"]["b2"]["edited_documents"] == 2
    assert first["arms"]["b2"]["edit_coverage"] == 1.0
    assert first["arms"]["b2"]["selected_edits"] == 2
    assert first["arms"]["b2"]["runtime_total_ms"] == 6
    assert first["incremental_tokens_saved"] == {"b2_vs_b0": 4, "b2_vs_b1": 2}
    assert report["runs"][0]["peak_memory"] == {"status": "unmeasured", "peak_rss_bytes": None}
    assert report["runs"][1]["peak_memory"] == {"status": "measured", "peak_rss_bytes": 123_456}
    assert report["runs"][2]["splits"]["held-out"]["incremental_tokens_saved"]["b2_vs_b1"] == 3
    out = tmp_path / "report"
    sizes.write_report(report, out, runs)
    assert json.loads((out / "report.json").read_text()) == report
    markdown = (out / "summary.md").read_text()
    assert "unmeasured" in markdown and "123456" in markdown
    assert "B2 vs B0" in markdown and "## held-out inputs" in markdown
    assert "Edit coverage" in markdown
    assert "not evaluated" in markdown
    with pytest.raises(ValueError, match="new directory"):
        sizes.write_report(report, out, runs)
    with pytest.raises(ValueError, match="outside source"):
        sizes.write_report(report, runs["10k"] / "nested", runs)


@pytest.mark.parametrize("change, message", [
    ({"docs": [("dev", "Different text.", "development"), DOCS[1]]}, "corpus text hashes"),
    ({"docs": [("dev", DOCS[0][1], "held-out"), DOCS[1]]}, "splits differ"),
    ({"config_change": {"max_candidates": 9}}, "settings apart"),
    ({"config_change": {"arms": ["u", "b0", "b1"]}}, "requires u/b0/b1/b2"),
    ({"vector_change": {"sha256": "b" * 64}}, "settings apart|source hash"),
    ({"vector_change": {"retained_rows": 49_999}}, "retained rows"),
    ({"input_tokens": 11}, "input token counts differ"),
    ({"measurements": {"peak_rss_bytes": 0}}, "positive measured"),
])
def test_rejects_incomparable_or_invalid_runs(tmp_path, change, message):
    runs = _runs(tmp_path, **{"50k": change})
    with pytest.raises(ValueError, match=message):
        sizes.build_report(runs)


def test_loader_rejects_tampering_and_missing_pairs(tmp_path):
    runs = _runs(tmp_path)
    path = runs["50k"]
    rows = (path / "results.jsonl").read_text().splitlines()
    (path / "results.jsonl").write_text("\n".join(rows[:-1]) + "\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        sizes.build_report(runs)
    (path / "checksums.json").unlink()
    with pytest.raises(ValueError, match="cover every input"):
        sizes.build_report(runs)


def test_cli_help_and_non_overwrite(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        sizes.main(["--help"])
    assert exc.value.code == 0
    assert "--run-10k" in capsys.readouterr().out
    runs = _runs(tmp_path)
    out = tmp_path / "out"
    args = ["--run-10k", str(runs["10k"]), "--run-50k", str(runs["50k"]),
            "--run-400k", str(runs["400k"]), "--out", str(out)]
    assert sizes.main(args) == 0
    before = (out / "report.json").read_bytes()
    assert sizes.main(args) == 2
    assert (out / "report.json").read_bytes() == before
