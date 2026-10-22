"""Human review imports preserve evidence, attribution, and pending states."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import shutil

import pytest

from eval import lexical_pilot as pilot
from eval import lexical_review as review
from promptcomp.lexical import PilotProvider


class WordCounter:
    name = "test:words-and-punctuation"

    def count(self, text):
        return len(re.findall(r"\w+|[^\w\s]", text))


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def refresh_checksums(run):
    """Permit internally inconsistent fixtures past byte-integrity validation."""
    (run / "checksums.json").write_text(json.dumps({
        name: hashlib.sha256((run / name).read_bytes()).hexdigest()
        for name in review.RUN_FILES
    }))


def refresh_exports(run):
    exports = pilot._review_exports(read_rows(run / "results.jsonl"))
    for name, rows in zip(("edit-review.jsonl", "prompt-review.jsonl", "review-key.jsonl"), exports):
        write_rows(run / name, rows)
    refresh_checksums(run)


def accept(row):
    return {**row, "meaning_preserved": True, "grammatical": True,
            "harmful_change": False, "reviewer": "test-reviewer"}


def arm(report, name="b0"):
    return next(row for row in report["arms"] if row["arm"] == name)


@pytest.fixture(scope="module")
def frozen_run(tmp_path_factory):
    run = tmp_path_factory.mktemp("lexical-review-source")
    documents = [
        pilot.PilotDocument("a", "In order to finish, they practiced.", "family-a"),
        pilot.PilotDocument("b", "They worked in order to finish. They prepared in order to improve.", "family-b"),
        pilot.PilotDocument("c", "Write a short answer.", "family-c"),
    ]
    provider = PilotProvider(phrases={"in order to": "to"})
    pilot.run_pilot(documents, {"u": None, "b0": provider, "b1": provider}, WordCounter(),
                    run, corpus_label="Authored review-import test fixture")
    return run


@pytest.fixture
def run(frozen_run, tmp_path):
    return Path(shutil.copytree(frozen_run, tmp_path / "run"))


def test_unanswered_source_stays_pending_and_unmodified(run):
    before = {path.name: path.read_bytes() for path in run.iterdir()}
    report = review.summarize_reviews(run)
    assert report["unique_edit_reviews"] == {
        "total": 3, "completed": 0, "partial": 0, "pending": 3,
        "accepted": 0, "rejected": 0, "harmful_completed": 0,
    }
    assert report["unique_prompt_reviews"]["total"] == 5
    assert report["unique_prompt_reviews"]["pending"] == 5
    assert arm(report)["changed_prompt_reviews"]["pending"] == 2
    assert arm(report)["unchanged_prompt_reviews"]["pending"] == 1
    assert arm(report)["tokens_saved_in_review_accepted_outputs"] == 0
    assert report["submitted"] == {"edit": None, "prompt": None}
    assert report["status"] == "annotation_summary_not_deployment_approval"
    assert before == {path.name: path.read_bytes() for path in run.iterdir()}


@pytest.mark.parametrize("kind", ["edit", "prompt"])
@pytest.mark.parametrize("bad_id", [None, 1, True, [], "", "unknown-id"])
def test_unknown_or_invalid_id_is_rejected(run, tmp_path, kind, bad_id):
    rows = read_rows(run / f"{kind}-review.jsonl")
    rows[0]["review_id"] = bad_id
    submitted = write_rows(tmp_path / "submitted.jsonl", rows[:1])
    with pytest.raises(ValueError, match="review_id"):
        review.summarize_reviews(run, **{f"{kind}_reviews": submitted})


@pytest.mark.parametrize("kind", ["edit", "prompt"])
def test_duplicate_id_even_with_identical_annotation_is_rejected(run, tmp_path, kind):
    row = read_rows(run / f"{kind}-review.jsonl")[0]
    submitted = write_rows(tmp_path / "submitted.jsonl", [row, row])
    with pytest.raises(ValueError, match="duplicate review_id"):
        review.summarize_reviews(run, **{f"{kind}_reviews": submitted})


@pytest.mark.parametrize("kind,field", [("edit", "original"), ("edit", "replacement"),
                                        ("edit", "sentence_after"), ("prompt", "original"),
                                        ("prompt", "suggested"), ("prompt", "doc_id")])
def test_immutable_content_cannot_be_changed(run, tmp_path, kind, field):
    row = read_rows(run / f"{kind}-review.jsonl")[0]
    row[field] += " tampered"
    submitted = write_rows(tmp_path / "submitted.jsonl", [row])
    with pytest.raises(ValueError, match="immutable review content changed"):
        review.summarize_reviews(run, **{f"{kind}_reviews": submitted})


def test_boolean_offset_does_not_equal_integer_offset(run, tmp_path):
    row = next(row for row in read_rows(run / "edit-review.jsonl") if row["span"]["start"] == 0)
    row["span"]["start"] = False
    submitted = write_rows(tmp_path / "submitted.jsonl", [row])
    with pytest.raises(ValueError, match="immutable review content changed"):
        review.summarize_reviews(run, edit_reviews=submitted)


@pytest.mark.parametrize("field,value,match", [
    ("meaning_preserved", 1, "true, false, or null"),
    ("grammatical", "true", "true, false, or null"),
    ("harmful_change", [], "true, false, or null"),
    ("reviewer", "   ", "nonblank string"),
    ("reviewer", False, "nonblank string"),
    ("reviewer", None, "reviewer identity"),
    ("notes", None, "notes must be a string"),
    ("notes", [], "notes must be a string"),
])
def test_annotation_types_and_identity_are_validated(run, tmp_path, field, value, match):
    row = accept(read_rows(run / "edit-review.jsonl")[0])
    row[field] = value
    submitted = write_rows(tmp_path / "submitted.jsonl", [row])
    with pytest.raises(ValueError, match=match):
        review.summarize_reviews(run, edit_reviews=submitted)


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_annotation_fields_must_match_export(run, tmp_path, mutation):
    row = read_rows(run / "prompt-review.jsonl")[0]
    if mutation == "missing":
        row.pop("notes")
    else:
        row["approved"] = True
    submitted = write_rows(tmp_path / "submitted.jsonl", [row])
    with pytest.raises(ValueError, match="fields differ"):
        review.summarize_reviews(run, prompt_reviews=submitted)


@pytest.mark.parametrize("payload,match", [
    ('{"review_id":"a","review_id":"b"}', "duplicate JSON field"),
    ('{"review_id":NaN}', "invalid JSON constant"),
    ('[]', "expected a JSON object"),
])
def test_malformed_json_is_rejected(run, tmp_path, payload, match):
    submitted = tmp_path / "submitted.jsonl"
    submitted.write_text(payload + "\n")
    with pytest.raises(ValueError, match=match):
        review.summarize_reviews(run, edit_reviews=submitted)


def test_partial_subset_preserves_pending_and_shares_attribution_across_arms(run, tmp_path):
    rows = read_rows(run / "edit-review.jsonl")
    submitted_rows = [accept(rows[0]), {**rows[1], "meaning_preserved": False, "reviewer": "test-reviewer"}]
    submitted = write_rows(tmp_path / "submitted.jsonl", submitted_rows)
    report = review.summarize_reviews(run, edit_reviews=submitted)
    assert report["unique_edit_reviews"] == {
        "total": 3, "completed": 1, "partial": 1, "pending": 1,
        "accepted": 1, "rejected": 0, "harmful_completed": 0,
    }
    for name in ("b0", "b1"):
        assert arm(report, name)["selected_edit_reviews"] == report["unique_edit_reviews"]
        assert arm(report, name)["changed_prompt_reviews"]["completed"] == 0
        assert arm(report, name)["tokens_saved_in_review_accepted_outputs"] == 0
    assert report["submitted"]["edit"]["sha256"] == hashlib.sha256(submitted.read_bytes()).hexdigest()
    assert report["unique_prompt_reviews"]["pending"] == 5


def test_accepted_rejected_candidate_is_only_candidate_evidence(run, tmp_path):
    results = read_rows(run / "results.jsonl")
    row = next(row for row in results if row["arm"] == "b0" and row["doc_id"] == "a")
    rejected = deepcopy(next(d for d in row["decisions"] if d["decision"] == "selected"))
    rejected.update(replacement="for", sentence_after="For finish, they practiced.",
                    decision="rejected", reason="test-grammar-rejection", token_delta=2)
    row["decisions"].append(rejected)
    write_rows(run / "results.jsonl", results)
    refresh_exports(run)
    annotation = next(row for row in read_rows(run / "edit-review.jsonl") if row["replacement"] == "for")
    submitted = write_rows(tmp_path / "submitted.jsonl", [accept(annotation)])
    report = review.summarize_reviews(run, edit_reviews=submitted)
    assert arm(report)["candidate_edit_reviews"]["accepted"] == 1
    assert arm(report)["selected_edit_reviews"]["accepted"] == 0
    assert arm(report)["changed_prompt_reviews"]["completed"] == 0
    assert arm(report)["tokens_saved_in_review_accepted_outputs"] == 0
    assert arm(report, "b1")["candidate_edit_reviews"]["accepted"] == 0


def test_changed_output_acceptance_counts_whole_output_against_all_input_tokens(run, tmp_path):
    rows = read_rows(run / "prompt-review.jsonl")
    changed = next(row for row in rows if row["doc_id"] == "a" and row["original"] != row["suggested"])
    unchanged = next(row for row in rows if row["doc_id"] == "c")
    submitted = write_rows(tmp_path / "submitted.jsonl", [accept(changed), accept(unchanged)])
    report = review.summarize_reviews(run, prompt_reviews=submitted)
    expected_before = sum(WordCounter().count(row["text"]) for row in read_rows(run / "inputs.jsonl"))
    for name in ("b0", "b1"):
        summary = arm(report, name)
        assert summary["changed_prompt_reviews"]["total"] == 2
        assert summary["changed_prompt_reviews"]["completed"] == 1
        assert summary["unchanged_prompt_reviews"]["total"] == 1
        assert summary["unchanged_prompt_reviews"]["accepted"] == 1
        assert summary["tokens_before"] == expected_before
        assert summary["tokens_saved_in_review_accepted_outputs"] == 2
        assert summary["review_accepted_savings_over_all_input_tokens"] == pytest.approx(2 / expected_before)
        assert summary["selected_edit_reviews"]["completed"] == 0
    assert arm(report, "u")["changed_prompt_reviews"]["total"] == 0
    assert arm(report, "u")["unchanged_prompt_reviews"]["total"] == 3
    assert arm(report, "u")["tokens_saved_in_review_accepted_outputs"] == 0


@pytest.mark.parametrize("field,value,harmful", [("meaning_preserved", False, 0),
                                                ("grammatical", False, 0),
                                                ("harmful_change", True, 1)])
def test_completed_rejections_are_not_pending_or_accepted(run, tmp_path, field, value, harmful):
    row = next(row for row in read_rows(run / "prompt-review.jsonl") if row["original"] != row["suggested"])
    annotation = {**accept(row), field: value}
    submitted = write_rows(tmp_path / "submitted.jsonl", [annotation])
    report = review.summarize_reviews(run, prompt_reviews=submitted)
    coverage = arm(report)["changed_prompt_reviews"]
    assert coverage["completed"] == 1 and coverage["rejected"] == 1
    assert coverage["accepted"] == 0 and coverage["harmful_completed"] == harmful
    assert arm(report)["tokens_saved_in_review_accepted_outputs"] == 0


def test_source_checksum_changes_are_rejected(run):
    with (run / "results.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="checksum mismatch: results.jsonl"):
        review.summarize_reviews(run)


@pytest.mark.parametrize("name", ["edit-review.jsonl", "prompt-review.jsonl", "review-key.jsonl"])
def test_rehashed_but_inconsistent_export_is_rejected(run, name):
    rows = read_rows(run / name)
    rows.pop()
    write_rows(run / name, rows)
    refresh_checksums(run)
    with pytest.raises(ValueError, match="exports do not match"):
        review.summarize_reviews(run)


def test_rehashed_duplicate_source_result_cannot_multiply_denominators(run):
    rows = read_rows(run / "results.jsonl")
    rows.append(deepcopy(rows[0]))
    write_rows(run / "results.jsonl", rows)
    refresh_exports(run)
    with pytest.raises(ValueError, match="duplicate"):
        review.summarize_reviews(run)


@pytest.mark.parametrize("change", ["missing", "extra"])
def test_checksum_set_must_be_exact(run, change):
    path = run / "checksums.json"
    checksums = json.loads(path.read_text())
    if change == "missing":
        checksums.pop("results.jsonl")
    else:
        checksums["../outside.json"] = "abc"
    path.write_text(json.dumps(checksums))
    with pytest.raises(ValueError, match="complete lexical-pilot-v1 checksum set"):
        review.summarize_reviews(run)


def test_unsupported_source_schema_is_rejected(run):
    path = run / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["schema_version"] = "unknown"
    path.write_text(json.dumps(manifest))
    refresh_checksums(run)
    with pytest.raises(ValueError, match="unsupported pilot schema"):
        review.summarize_reviews(run)


def test_report_is_reproducible_and_existing_output_is_never_overwritten(run, tmp_path):
    first = review.summarize_reviews(run)
    second = review.summarize_reviews(run)
    assert first["report_fingerprint"] == second["report_fingerprint"]
    out = tmp_path / "report"
    review.write_report(first, out)
    before = {path.name: path.read_bytes() for path in out.iterdir()}
    with pytest.raises(ValueError, match="new or empty directory"):
        review.write_report(second, out)
    assert before == {path.name: path.read_bytes() for path in out.iterdir()}
    assert "annotations" not in json.loads((out / "report.json").read_text())
    assert len(read_rows(out / "edit-annotations.jsonl")) == 3
    for name, digest in json.loads((out / "checksums.json").read_text()).items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest


def test_cli_writes_review_report_and_reports_bad_submission(run, tmp_path, capsys):
    out = tmp_path / "report"
    assert review.main(["--run", str(run), "--out", str(out)]) == 0
    assert "0/5 unique complete-prompt reviews completed" in capsys.readouterr().out
    assert (out / "summary.md").is_file()
    assert review.main(["--run", str(run), "--out", str(out)]) == 2
    assert "new or empty directory" in capsys.readouterr().err
    row = read_rows(run / "prompt-review.jsonl")[0]
    row["review_id"] = "unknown"
    submitted = write_rows(tmp_path / "submitted.jsonl", [row])
    bad_out = tmp_path / "bad-report"
    assert review.main(["--run", str(run), "--prompt-reviews", str(submitted), "--out", str(bad_out)]) == 2
    assert "unknown review_id" in capsys.readouterr().err
    assert not bad_out.exists()
