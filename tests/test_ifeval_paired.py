"""Paired IFEval evidence, original-gold scoring, and subset integrity."""
import hashlib
import json
from pathlib import Path
import re

import pytest

from eval import ifeval_paired as paired
from eval.lexical_pilot import PilotDocument, run_pilot
from promptcomp.lexical import PilotProvider


class WordCounter:
    name = "test:words"

    def count(self, value):
        return len(re.findall(r"\w+|[^\w\s]", value))


def jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


@pytest.fixture
def evidence(tmp_path):
    run = tmp_path / "run"
    originals = {
        "ifeval-101": "They worked in order to finish. Say alpha without commas.",
        "ifeval-102": "Say beta without commas.",
        "ifeval-103": "They worked in order to finish. Say gamma without commas.",
    }
    docs = [PilotDocument(doc_id, prompt, "family", "held_out" if doc_id.endswith("3")
                          else "development") for doc_id, prompt in originals.items()]
    run_pilot(docs, {"u": None,
                     "b0": PilotProvider(phrases={"in order to": "to"}),
                     "b2": None},
              WordCounter(), run, corpus_label="IFEval test",
              configuration={"arms": ["u", "b0", "b2"]})
    results = [json.loads(line) for line in (run / "results.jsonl").read_text().splitlines()]
    gold = tmp_path / "gold.jsonl"
    source_sha = "a" * 64
    gold_rows = []
    for doc_id, prompt in originals.items():
        record = {"key": int(doc_id.split("-")[1]), "prompt": prompt,
                  "instruction_id_list": ["no_comma"], "kwargs": [{"marker": "original"}]}
        digest = hashlib.sha256(json.dumps(record, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        gold_rows.append({"id": doc_id, **record, "source_sha256": source_sha,
                          "record_sha256": digest})
    jsonl(gold, gold_rows)
    (tmp_path / "provenance.json").write_text(json.dumps({
        "schema_version": "ifeval-frozen-v1", "source_sha256": source_sha,
        "records": len(gold_rows),
        "file_sha256": {"gold.jsonl": hashlib.sha256(gold.read_bytes()).hexdigest()}}))
    package = tmp_path / "official" / "instruction_following_eval"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "evaluation_lib.py").write_text('''
from collections import namedtuple
InputExample = namedtuple("InputExample", "key prompt instruction_id_list kwargs")
Output = namedtuple("Output", "follow_all_instructions follow_instruction_list")
def _score(inp, answers, loose):
    assert "Say " in inp.prompt
    assert inp.kwargs == [{"marker": "original"}]
    assert inp.prompt in answers and len(answers) == 1
    assert "in order to" in inp.prompt if inp.key in (101, 103) else True
    response = answers[inp.prompt]
    passed = "," not in response and ("alpha" in response if inp.key == 101 else
                                        "beta" in response if inp.key == 102 else "gamma" in response)
    if loose and response.strip().lower() == "almost":
        passed = True
    return Output(passed, [passed])
def test_instruction_following_strict(inp, answers):
    return _score(inp, answers, False)
def test_instruction_following_loose(inp, answers):
    return _score(inp, answers, True)
''')
    return run, gold, tmp_path / "official", results


def responses_for(results, doc_ids):
    return [{"doc_id": row["doc_id"], "arm": row["arm"],
             "prompt_sha256": hashlib.sha256(row["suggested"].encode()).hexdigest(),
             "response": ("alpha" if row["arm"] == "u" else "alpha,")
                         if row["doc_id"].endswith("1") else
                         "beta" if row["doc_id"].endswith("2") else
                         ("gamma," if row["arm"] == "u" else "gamma"),
             "model_id": "local/test", "model_revision": "abc123",
             "decoding": {"temperature": 0, "max_new_tokens": 32},
             "model_input_tokens": 20 if row["arm"] == "u" else 16,
             "output_tokens": 2}
            for row in results if row["doc_id"] in doc_ids]


def test_cli_scores_original_gold_and_paired_changed_slice(evidence, tmp_path):
    run, gold, official, results = evidence
    responses = tmp_path / "responses.jsonl"
    jsonl(responses, responses_for(results, {"ifeval-101", "ifeval-102"}))
    out = tmp_path / "out"
    assert paired.main(["--run", str(run), "--gold", str(gold), "--responses", str(responses),
                        "--evaluator-path", str(official), "--out", str(out)]) == 0
    report = json.loads((out / "report.json").read_text())
    assert report["selected_documents"] == 2
    assert report["pilot_documents"] == 3
    assert report["model"] == {"model_id": "local/test", "model_revision": "abc123",
                               "decoding": {"temperature": 0, "max_new_tokens": 32}}
    assert report["excluded_truncated_count"] is None
    assert "truncation is unverified" in report["limitations"][-1]
    b0 = report["arms"][1]
    assert b0["model_input_tokens_saved"] == 8
    assert b0["slices"]["all_selected"]["strict"]["prompt_passes"] == 1
    changed = b0["slices"]["changed_only"]["strict"]
    assert (changed["prompt_total"], changed["prompt_passes"],
            changed["regressions_vs_u"], changed["recoveries_vs_u"]) == (1, 0, 1, 0)
    assert report["arms"][0]["slices"]["changed_only"]["strict"]["prompt_total"] == 1
    assert report["arms"][2]["slices"]["changed_only"]["strict"]["prompt_total"] == 1
    scored = [json.loads(line) for line in (out / "scores.jsonl").read_text().splitlines()]
    assert {row["doc_id"] for row in scored if row["changed_by_any"]} == {"ifeval-101"}
    assert next(row for row in scored if row["doc_id"] == "ifeval-101" and row["arm"] == "b2")["changed"] is False
    assert changed["instruction_total"] == 1
    assert set(report["splits"]) == {"development"}
    assert len((out / "scores.jsonl").read_text().splitlines()) == 6
    assert paired.main(["--run", str(run), "--gold", str(gold), "--responses", str(responses),
                        "--evaluator-path", str(official), "--out", str(out)]) == 2


def test_changing_complete_subset_changes_denominators_and_recoveries(evidence, tmp_path):
    run, gold, official, results = evidence
    responses = tmp_path / "responses.jsonl"
    jsonl(responses, responses_for(results, {"ifeval-103"}))
    report, scores = paired.evaluate(run, gold, responses, official)
    assert report["selected_documents"] == 1
    assert set(report["splits"]) == {"held_out"}
    changed = report["arms"][1]["slices"]["changed_only"]["strict"]
    assert (changed["prompt_total"], changed["recoveries_vs_u"],
            changed["regressions_vs_u"]) == (1, 1, 0)
    assert len(scores) == 3
    assert {row["changed_by_any"] for row in scores} == {True}
    assert next(row for row in scores if row["arm"] == "u")["changed"] is False


@pytest.mark.parametrize("mutate, message", [
    (lambda rows: rows.pop(), "missing response arms"),
    (lambda rows: rows.append(dict(rows[0])), "duplicate response pair"),
    (lambda rows: rows[0].update(prompt_sha256="0" * 64), "prompt SHA mismatch"),
    (lambda rows: rows[0].update(arm="extra"), "extra arm"),
    (lambda rows: rows[0].update(model_revision="other"), "model identity"),
    (lambda rows: rows[0].update(model_input_tokens=-1), "nonnegative integer"),
])
def test_rejects_invalid_response_sets(evidence, tmp_path, mutate, message):
    run, gold, official, results = evidence
    rows = responses_for(results, {"ifeval-101"})
    mutate(rows)
    responses = tmp_path / "responses.jsonl"
    jsonl(responses, rows)
    with pytest.raises(ValueError, match=message):
        paired.evaluate(run, gold, responses, official)


def test_rejects_rewritten_gold_even_when_response_hash_matches(evidence, tmp_path):
    run, gold, official, results = evidence
    rows = [json.loads(line) for line in gold.read_text().splitlines()]
    rows[0]["prompt"] = next(row["suggested"] for row in results
                             if row["doc_id"] == "ifeval-101" and row["arm"] == "b0")
    jsonl(gold, rows)
    responses = tmp_path / "responses.jsonl"
    jsonl(responses, responses_for(results, {"ifeval-101"}))
    with pytest.raises(ValueError, match="gold file SHA mismatch"):
        paired.evaluate(run, gold, responses, official)


def test_rejects_tampered_constraints_even_with_recomputed_file_sha(evidence, tmp_path):
    run, gold, official, results = evidence
    rows = [json.loads(line) for line in gold.read_text().splitlines()]
    rows[0]["kwargs"] = [{"marker": "tampered"}]
    jsonl(gold, rows)
    provenance_path = gold.parent / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["file_sha256"]["gold.jsonl"] = hashlib.sha256(gold.read_bytes()).hexdigest()
    provenance_path.write_text(json.dumps(provenance))
    responses = tmp_path / "responses.jsonl"
    jsonl(responses, responses_for(results, {"ifeval-101"}))
    with pytest.raises(ValueError, match="gold record or source SHA mismatch"):
        paired.evaluate(run, gold, responses, official)


def test_generation_manifest_binds_response_file_and_run(evidence, tmp_path):
    run, gold, official, results = evidence
    responses = tmp_path / "responses.jsonl"
    jsonl(responses, responses_for(results, {"ifeval-101"}))
    fingerprint = json.loads((run / "manifest.json").read_text())["run_fingerprint"]
    manifest = {"schema_version": "ifeval-generate-v1", "source_run_fingerprint": fingerprint,
                "files": {"responses.jsonl": hashlib.sha256(responses.read_bytes()).hexdigest()},
                "rows": 3, "selected_doc_ids": ["ifeval-101"], "arms": ["u", "b0", "b2"],
                "model": {"id": "local/test", "revision": "abc123"},
                "settings": {"decoding": {"temperature": 0, "max_new_tokens": 32}},
                "truncation": [{"doc_id": "ifeval-101", "arm": arm, "truncated": False}
                               for arm in ("u", "b0", "b2")], "truncated_rows": 0}
    generation = tmp_path / "manifest.json"
    generation.write_text(json.dumps(manifest))
    report, _ = paired.evaluate(run, gold, responses, official,
                                generation_manifest=generation)
    assert report["source_hashes"]["generation_manifest_sha256"]
    manifest["source_run_fingerprint"] = "wrong"
    generation.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="generation manifest does not match"):
        paired.evaluate(run, gold, responses, official, generation_manifest=generation)


def test_generator_selected_u_and_b2_subset_scores_as_pair(evidence, tmp_path):
    run, gold, official, results = evidence
    responses = tmp_path / "responses.jsonl"
    rows = [r for r in responses_for(results, {"ifeval-101"}) if r["arm"] in ("u", "b2")]
    jsonl(responses, rows)
    fingerprint = json.loads((run / "manifest.json").read_text())["run_fingerprint"]
    generation = tmp_path / "manifest.json"
    generation.write_text(json.dumps({
        "schema_version": "ifeval-generate-v1", "source_run_fingerprint": fingerprint,
        "files": {"responses.jsonl": hashlib.sha256(responses.read_bytes()).hexdigest()},
        "rows": 2, "selected_doc_ids": ["ifeval-101"], "arms": ["u", "b2"],
        "model": {"id": "local/test", "revision": "abc123"},
        "settings": {"decoding": {"temperature": 0, "max_new_tokens": 32}},
        "truncation": [{"doc_id": "ifeval-101", "arm": arm, "truncated": False}
                       for arm in ("u", "b2")], "truncated_rows": 0}))
    report, scores = paired.evaluate(run, gold, responses, official,
                                     generation_manifest=generation)
    assert [a["arm"] for a in report["arms"]] == ["u", "b2"]
    assert len(scores) == 2
    assert report["arms"][1]["slices"]["changed_only"]["strict"]["prompt_total"] == 0
    record = json.loads(generation.read_text())
    record["settings"]["changed_arm"] = "b0"
    generation.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="changed_arm does not match"):
        paired.evaluate(run, gold, responses, official, generation_manifest=generation)
    record["arms"] = ["u"]
    generation.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="comparison arm"):
        paired.evaluate(run, gold, responses, official, generation_manifest=generation)


def test_truncation_excludes_whole_doc_and_all_excluded_fails(evidence, tmp_path):
    run, gold, official, results = evidence
    responses = tmp_path / "responses.jsonl"
    rows = responses_for(results, {"ifeval-101", "ifeval-102"})
    for row in rows:
        row["output_tokens"] = 32
    jsonl(responses, rows)
    fingerprint = json.loads((run / "manifest.json").read_text())["run_fingerprint"]
    manifest = {"schema_version": "ifeval-generate-v1", "source_run_fingerprint": fingerprint,
                "files": {"responses.jsonl": hashlib.sha256(responses.read_bytes()).hexdigest()},
                "rows": 6, "selected_doc_ids": ["ifeval-101", "ifeval-102"],
                "arms": ["u", "b0", "b2"],
                "model": {"id": "local/test", "revision": "abc123"},
                "settings": {"decoding": {"temperature": 0, "max_new_tokens": 32}},
                "truncation": [{"doc_id": doc, "arm": arm,
                                "truncated": doc == "ifeval-101" and arm == "b2"}
                               for doc in ("ifeval-101", "ifeval-102")
                               for arm in ("u", "b0", "b2")], "truncated_rows": 1}
    generation = tmp_path / "manifest.json"
    generation.write_text(json.dumps(manifest))
    out = tmp_path / "out"
    assert paired.main(["--run", str(run), "--gold", str(gold),
                        "--responses", str(responses), "--evaluator-path", str(official),
                        "--generation-manifest", str(generation), "--out", str(out)]) == 0
    report = json.loads((out / "report.json").read_text())
    assert report["response_documents"] == 2
    assert report["selected_documents"] == 1
    assert report["excluded_truncated_doc_ids"] == ["ifeval-101"]
    assert report["excluded_truncated_count"] == 1
    assert all(arm["slices"]["changed_only"]["strict"]["prompt_total"] == 0
               for arm in report["arms"])
    assert {json.loads(line)["doc_id"] for line in (out / "scores.jsonl").read_text().splitlines()} == {"ifeval-102"}
    for row in manifest["truncation"]:
        if row["doc_id"] == "ifeval-102":
            row["truncated"] = True
    manifest["truncated_rows"] = 4
    generation.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="all selected documents were truncated"):
        paired.evaluate(run, gold, responses, official, generation_manifest=generation)


def test_rejects_incomplete_truncation_list(evidence, tmp_path):
    run, gold, official, results = evidence
    responses = tmp_path / "responses.jsonl"
    jsonl(responses, responses_for(results, {"ifeval-101"}))
    fingerprint = json.loads((run / "manifest.json").read_text())["run_fingerprint"]
    generation = tmp_path / "manifest.json"
    generation.write_text(json.dumps({
        "schema_version": "ifeval-generate-v1", "source_run_fingerprint": fingerprint,
        "files": {"responses.jsonl": hashlib.sha256(responses.read_bytes()).hexdigest()},
        "rows": 3, "selected_doc_ids": ["ifeval-101"], "arms": ["u", "b0", "b2"],
        "model": {"id": "local/test", "revision": "abc123"},
        "settings": {"decoding": {"temperature": 0, "max_new_tokens": 32}},
        "truncation": [{"doc_id": "ifeval-101", "arm": "u", "truncated": False}],
        "truncated_rows": 0}))
    with pytest.raises(ValueError, match="truncation list does not cover"):
        paired.evaluate(run, gold, responses, official, generation_manifest=generation)


def test_requires_pilot_checksums_and_unchanged_baseline(evidence, tmp_path, monkeypatch):
    run, gold, official, results = evidence
    responses = tmp_path / "responses.jsonl"
    jsonl(responses, responses_for(results, {"ifeval-101"}))
    original_source = paired._source
    def altered_source(path):
        manifest, inputs, rows = original_source(path)
        for row in rows:
            if row["doc_id"] == "ifeval-101" and row["arm"] == "u":
                row["suggested"] = "modified baseline"
        return manifest, inputs, rows
    monkeypatch.setattr(paired, "_source", altered_source)
    with pytest.raises(ValueError, match="baseline arm u modified"):
        paired.evaluate(run, gold, responses, official)
    monkeypatch.setattr(paired, "_source", original_source)
    (run / "checksums.json").unlink()
    with pytest.raises(ValueError, match="checksums.json is required"):
        paired.evaluate(run, gold, responses, official)
