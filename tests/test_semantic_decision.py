"""Paired holdout reporting from fingerprinted, unreviewed pilot evidence."""
import hashlib
import json

import pytest

from eval import lexical_pilot as pilot
from eval import semantic_decision as decision


DOCS = [
    ("dev", "Development text.", "dev-group", "development"),
    ("h1", "Held out one.", "group-a", "held-out"),
    ("h2", "Held out two.", "group-a", "held-out"),
    ("h3", "Held out three.", "group-b", "held-out"),
]
SAVED = {
    "dev": {"u": 0, "b0": 1, "b1": 2, "b2": 3},
    "h1": {"u": 0, "b0": 1, "b1": 3, "b2": 4},
    "h2": {"u": 0, "b0": 2, "b1": 2, "b2": 1},
    "h3": {"u": 0, "b0": 0, "b1": 1, "b2": 4},
}


def _write_run(path, docs=DOCS, *, arms=decision.ARMS, saved=SAVED):
    path.mkdir()
    inputs = [{"id": doc_id, "group": group, "split": split,
               "sha256": hashlib.sha256(text.encode()).hexdigest()}
              for doc_id, text, group, split in docs]
    identity = {"schema_version": "lexical-pilot-v1", "inputs": inputs, "counter": "test-counter",
                "providers": {}, "configuration": {"arms": list(arms)},
                "versions": {}, "software_sha256": {}}
    manifest = {**identity, "run_fingerprint": pilot._digest(identity), "corpus_label": "test"}
    rows = []
    for doc_id, original, group, split in docs:
        for arm in arms:
            saving = saved[doc_id][arm]
            rows.append({"doc_id": doc_id, "group": group, "split": split, "arm": arm,
                         "original": original, "suggested": original if not saving else f"{original} [{arm}]",
                         "tokens_before": 10, "tokens_after": 10 - saving, "token_counter": "test-counter",
                         "edits": [{}] if saving else [], "timings_ms": {"total": {"u": 0, "b0": 2,
                                                                                "b1": 4, "b2": 7}[arm]}})
    summary = []
    for arm in arms:
        selected = [row for row in rows if row["arm"] == arm]
        summary.append({"arm": arm, "documents": len(selected),
                        "tokens_before": sum(row["tokens_before"] for row in selected),
                        "tokens_after": sum(row["tokens_after"] for row in selected),
                        "tokens_saved": sum(row["tokens_before"] - row["tokens_after"] for row in selected),
                        "changed_documents": sum(row["original"] != row["suggested"] for row in selected)})
    (path / "manifest.json").write_text(json.dumps(manifest))
    (path / "results.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    (path / "summary.json").write_text(json.dumps(summary))
    (path / "checksums.json").write_text(json.dumps({name: hashlib.sha256((path / name).read_bytes()).hexdigest()
                                                       for name in ("manifest.json", "results.jsonl", "summary.json")}))
    return path


def test_paired_deltas_and_held_out_group_bootstrap(tmp_path):
    run = _write_run(tmp_path / "run")
    report = decision.build_report(run)
    assert report["quality"] == "unreviewed"
    assert report["net_economics"] == "unknown"
    assert report["decision"] == "no_production_go_no_go_claim"
    held = report["splits"]["held-out"]
    assert held["documents"] == 3 and held["groups"] == 2
    assert [(row["arm"], row["all_input_tokens"], row["tokens_saved"],
             row["changed_documents"], row["selected_edits"]) for row in held["arms"]] == [
                 ("u", 30, 0, 0, 0), ("b0", 30, 3, 2, 2),
                 ("b1", 30, 6, 3, 3), ("b2", 30, 9, 3, 3)]
    assert held["arms"][2]["p50_ms"] == 4
    pairs = {(row["comparison"], row["doc_id"]): row for row in report["paired_documents"]}
    assert [pairs["b1-b0", doc]["tokens_saved_delta"] for doc in ("h1", "h2", "h3")] == [2, 0, 1]
    assert [pairs["b2-b1", doc]["tokens_saved_delta"] for doc in ("h1", "h2", "h3")] == [1, -1, 3]
    assert [pairs["b2-b0", doc]["tokens_saved_delta"] for doc in ("h1", "h2", "h3")] == [3, -1, 4]
    assert "b2-b0" in report["held_out_uncertainty"]
    assert pairs["b1-b0", "h1"]["tokens_after_delta"] == -2
    uncertainty = report["held_out_uncertainty"]["b1-b0"]
    assert uncertainty["mean_document_delta"] == 1
    assert uncertainty["groups"] == 2 and uncertainty["documents"] == 3
    assert uncertainty["group_deltas"] == [
        {"group": "group-a", "documents": 2, "tokens_saved_delta": 2, "mean_document_delta": 1},
        {"group": "group-b", "documents": 1, "tokens_saved_delta": 1, "mean_document_delta": 1},
    ]
    assert uncertainty["interval_95"] == [1, 1]
    assert decision.build_report(run) == report


def test_missing_pairs_and_missing_arm_are_rejected(tmp_path):
    run = _write_run(tmp_path / "run")
    rows = (run / "results.jsonl").read_text().splitlines()
    (run / "results.jsonl").write_text("\n".join(rows[:-1]) + "\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        decision.build_report(run)
    (run / "checksums.json").unlink()
    with pytest.raises(ValueError, match="cover every input"):
        decision.build_report(run)
    short = _write_run(tmp_path / "short", arms=("u", "b0", "b1"))
    with pytest.raises(ValueError, match="u/b0/b1/b2"):
        decision.build_report(short)


def test_category_prefix_reports_each_split_without_family_leakage(tmp_path):
    docs = [
        ("dev", "Development text.", "writing:dev-family", "development"),
        ("h1", "Held out one.", "writing:holdout-family", "held-out"),
        ("h2", "Held out two.", "news:holdout-family", "held-out"),
    ]
    run = _write_run(tmp_path / "run", docs=docs)
    report = decision.build_report(run)
    development = report["splits"]["development"]["categories"]
    held = report["splits"]["held-out"]["categories"]
    assert [row["category"] for row in development] == ["writing"]
    assert [row["category"] for row in held] == ["news", "writing"]
    assert held[1]["comparisons"]["b1-b0"]["tokens_saved_delta"] == 2
    assert held[1]["comparisons"]["b1-b0"]["mean_document_delta"] == 2
    assert report["paired_documents"][0]["category"] in {"news", "writing"}
    out = tmp_path / "report"
    decision.write_report(report, out)
    markdown = (out / "summary.md").read_text()
    assert "## Development" in markdown and "## Held-Out" in markdown
    assert "| writing |" in markdown and "| news |" in markdown


@pytest.mark.parametrize("docs, message", [
    ([DOCS[0]], "development and held-out"),
    ([DOCS[1]], "development and held-out"),
    ([DOCS[0], ("h", "  DEVELOPMENT  TEXT. ", "other", "held-out")], "duplicate normalized text"),
    ([DOCS[0], ("h", "Different text.", "dev-group", "held-out")], "same group across splits"),
])
def test_missing_splits_and_leakage_rejected(tmp_path, docs, message):
    saved = {doc[0]: {arm: 0 for arm in decision.ARMS} for doc in docs}
    run = _write_run(tmp_path / "run", docs=docs, saved=saved)
    with pytest.raises(ValueError, match=message):
        decision.build_report(run)


def test_cli_writes_json_and_markdown_to_new_directory_only(tmp_path):
    run = _write_run(tmp_path / "run")
    out = tmp_path / "report"
    assert decision.main(["--run", str(run), "--out", str(out)]) == 0
    payload = json.loads((out / "report.json").read_text())
    markdown = (out / "summary.md").read_text()
    assert payload["schema_version"] == "semantic-decision-v1"
    assert "unreviewed" in markdown and "unknown" in markdown
    assert "No production go/no-go claim" in markdown
    assert decision.main(["--run", str(run), "--out", str(out)]) == 2
    assert json.loads((out / "report.json").read_text()) == payload
    assert decision.main(["--run", str(run), "--out", str(run / "nested")]) == 2
    assert not (run / "nested").exists()
