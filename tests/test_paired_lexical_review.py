"""Paired prompt reviews count only complete, independent judgments."""
import json
import re

import pytest

from eval import lexical_pilot as pilot
from eval import paired_lexical_review as paired
from promptcomp.lexical import PilotProvider


class WordCounter:
    name = "test:words-and-punctuation"

    def count(self, text):
        return len(re.findall(r"\w+|[^\w\s]", text))


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def packet(path, data):
    path.write_text("".join(json.dumps(row) + "\n" for row in data), encoding="utf-8")
    return path


def judged(row, reviewer, **changes):
    return {**row, "meaning_preserved": True, "grammatical": True,
            "harmful_change": False, "reviewer": reviewer, **changes}


@pytest.fixture
def run(tmp_path):
    source = tmp_path / "run"
    documents = [
        pilot.PilotDocument("a", "In order to finish, they practiced.", "family-a"),
        pilot.PilotDocument("b", "They worked in order to finish. They prepared in order to improve.", "family-b"),
        pilot.PilotDocument("c", "Write a short answer.", "family-c"),
    ]
    provider = PilotProvider(phrases={"in order to": "to"})
    pilot.run_pilot(documents, {"u": None, "b0": provider, "b1": provider}, WordCounter(),
                    source, corpus_label="Paired-review test fixture")
    return source


def changed(run):
    return [row for row in rows(run / "prompt-review.jsonl")
            if row["original"] != row["suggested"]]


def test_sparse_pairs_count_unique_changed_ids_and_conservative_arm_savings(run, tmp_path):
    first, second = changed(run)
    unchanged = next(row for row in rows(run / "prompt-review.jsonl")
                     if row["doc_id"] == "c")
    a = packet(tmp_path / "a.jsonl", [judged(first, "A"), judged(second, "A"),
                                      judged(unchanged, "A")])
    b = packet(tmp_path / "b.jsonl", [judged(first, "B"),
                                      judged(second, "B", harmful_change=True),
                                      judged(unchanged, "B")])
    report = paired.summarize_paired_reviews(run, a, b)
    counts = report["changed_review_ids"]
    assert counts == {"total": 2, "pending": 0, "partial": 0, "both_complete": 2,
                      "agreement": 1, "disagreement": 1, "any_harmful_true": 1,
                      "jointly_accepted": 1}
    assert report["reviewer_coverage"]["a"]["completed"] == 2
    assert len(report["disagreements"]) == 1
    assert report["disagreements"][0]["review_id"] == second["review_id"]
    assert {"disagreement", "harmful_flag", "both_complete_rejected"} <= set(report["triage"][0]["reasons"])
    assert not any("arm" in key or "source" in key for key in report["disagreements"][0])
    before = sum(WordCounter().count(row["text"]) for row in rows(run / "inputs.jsonl"))
    accepted_saving = next(row["tokens_before"] - row["tokens_after"]
                           for row in rows(run / "results.jsonl")
                           if row["arm"] == "b0" and
                           pilot._digest({"doc_id": row["doc_id"], "original": row["original"],
                                          "suggested": row["suggested"]}) == first["review_id"])
    for arm in report["arms"]:
        assert arm["tokens_before"] == before
        if arm["arm"] in ("b0", "b1"):
            assert arm["changed_review_ids"] == counts
            assert arm["tokens_saved_in_jointly_accepted_changed_outputs"] == accepted_saving
            assert arm["jointly_accepted_savings_over_all_input_tokens"] == pytest.approx(accepted_saving / before)
        else:
            assert arm["changed_review_ids"]["total"] == 0
            assert arm["tokens_saved_in_jointly_accepted_changed_outputs"] == 0

    out = tmp_path / "report"
    paired.write_report(report, out, run_dir=run)
    assert rows(out / "disagreements.jsonl") == report["disagreements"]
    assert rows(out / "triage.jsonl") == report["triage"]
    assert "disagreements" not in json.loads((out / "report.json").read_text())
    assert "not passes or deployment approval" in (out / "summary.md").read_text()
    with pytest.raises(ValueError, match="new directory"):
        paired.write_report(report, out, run_dir=run)


def test_pending_partial_and_unilateral_acceptance_never_pass(run, tmp_path):
    first, second = changed(run)
    a = packet(tmp_path / "a.jsonl", [judged(first, "A"),
                                      {**second, "meaning_preserved": True, "reviewer": "A"}])
    b = packet(tmp_path / "b.jsonl", [])
    report = paired.summarize_paired_reviews(run, a, b)
    assert report["changed_review_ids"] == {
        "total": 2, "pending": 0, "partial": 2, "both_complete": 0,
        "agreement": 0, "disagreement": 0, "any_harmful_true": 0, "jointly_accepted": 0,
    }
    assert report["reviewer_coverage"]["a"] == {"pending": 0, "partial": 1, "completed": 1}
    assert report["reviewer_coverage"]["b"] == {"pending": 2, "partial": 0, "completed": 0}
    assert all(arm["tokens_saved_in_jointly_accepted_changed_outputs"] == 0
               for arm in report["arms"])
    packet(a, [])
    assert paired.summarize_paired_reviews(run, a, b)["changed_review_ids"]["pending"] == 2


@pytest.mark.parametrize("mutation,match", [
    ("same_identity", "identities must differ"),
    ("inconsistent_identity", "inconsistent reviewer identities"),
    ("tamper", "immutable review content changed"),
    ("duplicate", "duplicate review_id"),
    ("unknown", "unknown review_id"),
    ("integer_judgment", "true, false, or null"),
    ("missing_identity", "reviewer identity"),
])
def test_rejects_invalid_packets(run, tmp_path, mutation, match):
    first, second = changed(run)
    a_rows = [judged(first, "A")]
    b_rows = [judged(first, "B")]
    if mutation == "same_identity":
        b_rows[0]["reviewer"] = "A"
    elif mutation == "inconsistent_identity":
        a_rows.append(judged(second, "C"))
    elif mutation == "tamper":
        b_rows[0]["suggested"] += " altered"
    elif mutation == "duplicate":
        b_rows.append(b_rows[0])
    elif mutation == "unknown":
        b_rows[0]["review_id"] = "unknown"
    elif mutation == "integer_judgment":
        b_rows[0]["harmful_change"] = 1
    elif mutation == "missing_identity":
        b_rows[0]["reviewer"] = None
    a = packet(tmp_path / "a.jsonl", a_rows)
    b = packet(tmp_path / "b.jsonl", b_rows)
    with pytest.raises(ValueError, match=match):
        paired.summarize_paired_reviews(run, a, b)


def test_source_checksum_and_output_guard_and_cli(run, tmp_path):
    a = packet(tmp_path / "a.jsonl", [])
    b = packet(tmp_path / "b.jsonl", [])
    out = tmp_path / "report"
    assert paired.main(["--run", str(run), "--reviewer-a", str(a),
                        "--reviewer-b", str(b), "--out", str(out)]) == 0
    assert json.loads((out / "report.json").read_text())["changed_review_ids"]["pending"] == 2
    assert paired.main(["--run", str(run), "--reviewer-a", str(a),
                        "--reviewer-b", str(b), "--out", str(out)]) == 2
    report = paired.summarize_paired_reviews(run, a, b)
    with pytest.raises(ValueError, match="inside the frozen run"):
        paired.write_report(report, run / "nested", run_dir=run)
    with (run / "results.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="checksum mismatch: results.jsonl"):
        paired.summarize_paired_reviews(run, a, b)


def test_no_same_packet_path_and_harmful_partial_is_flagged(run, tmp_path):
    first = changed(run)[0]
    a = packet(tmp_path / "a.jsonl", [{**first, "harmful_change": True, "reviewer": "A"}])
    b = packet(tmp_path / "b.jsonl", [])
    with pytest.raises(ValueError, match="separate files"):
        paired.summarize_paired_reviews(run, a, a)
    counts = paired.summarize_paired_reviews(run, a, b)["changed_review_ids"]
    assert counts["any_harmful_true"] == 1
    assert counts["jointly_accepted"] == 0
    triage = paired.summarize_paired_reviews(run, a, b)["triage"]
    assert triage[0]["reasons"] == ["harmful_flag"]
    assert triage[0]["reviewer_a"]["harmful_change"] is True


@pytest.mark.parametrize("same_name", ["alice", " Alice ", "Ａｌｉｃｅ"])
def test_reviewer_identity_uses_case_whitespace_and_unicode_normalization(run, tmp_path, same_name):
    first = changed(run)[0]
    a = packet(tmp_path / "a.jsonl", [judged(first, "Alice")])
    b = packet(tmp_path / "b.jsonl", [judged(first, same_name)])
    with pytest.raises(ValueError, match="identities must differ"):
        paired.summarize_paired_reviews(run, a, b)


def test_agreed_harmful_change_is_triaged_with_notes(run, tmp_path):
    first = changed(run)[0]
    a = packet(tmp_path / "a.jsonl", [judged(first, "A", harmful_change=True, notes="unsafe")])
    b = packet(tmp_path / "b.jsonl", [judged(first, "B", harmful_change=True, notes="meaning shifts")])
    report = paired.summarize_paired_reviews(run, a, b)
    assert report["changed_review_ids"]["agreement"] == 1
    assert report["changed_review_ids"]["jointly_accepted"] == 0
    assert report["disagreements"] == []
    assert len(report["triage"]) == 1
    assert report["triage"][0]["reasons"] == ["harmful_flag", "both_complete_rejected"]
    assert report["triage"][0]["reviewer_a"]["notes"] == "unsafe"
