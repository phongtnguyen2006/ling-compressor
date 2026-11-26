"""Reviewer packets are blind, blank, reproducible, and never overwrite output."""
import hashlib
import json
from pathlib import Path
import re
import shutil

import pytest

from eval import lexical_pilot as pilot
from eval import lexical_review as review
from eval import lexical_review_packet as packet
from promptcomp.lexical import PilotProvider


class WordCounter:
    name = "test:words-and-punctuation"

    def count(self, text):
        return len(re.findall(r"\w+|[^\w\s]", text))


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def make_run(path, documents):
    provider = PilotProvider(phrases={"in order to": "to"})
    pilot.run_pilot(documents, {"u": None, "b0": provider, "b1": provider}, WordCounter(),
                    path, corpus_label="Authored packet test fixture")
    return path


@pytest.fixture(scope="module")
def frozen_run(tmp_path_factory):
    docs = [pilot.PilotDocument("a", "In order to finish, they practiced.", "f-a"),
            pilot.PilotDocument("b", "They worked in order to finish. They prepared in order to improve.", "f-b"),
            pilot.PilotDocument("c", "Write a short answer.", "f-c"),
            pilot.PilotDocument("d", "Another plain prompt.", "f-d"),
            pilot.PilotDocument("e", "Third plain prompt here.", "f-e")]
    return make_run(tmp_path_factory.mktemp("packet-source") / "run", docs)


@pytest.fixture
def run(frozen_run, tmp_path):
    return Path(shutil.copytree(frozen_run, tmp_path / "run"))


def test_contains_all_changed_and_default_controls_blank(run, tmp_path):
    before = {p.name: p.read_bytes() for p in run.iterdir()}
    meta = packet.build_packet(run, tmp_path / "out")
    source = read_rows(run / "prompt-review.jsonl")
    changed = {r["review_id"] for r in source if r["original"] != r["suggested"]}
    unchanged = {r["review_id"] for r in source if r["original"] == r["suggested"]
                 and r["doc_id"] in "cde"}
    assert len(changed) == 2 and len(unchanged) == 3
    assert meta["counts"]["changed"] == 2 and meta["counts"]["controls"] == 2
    by_id = {r["review_id"]: r for r in source}
    for name in ("reviewer-a.jsonl", "reviewer-b.jsonl"):
        rows = read_rows(tmp_path / "out" / name)
        ids = [r["review_id"] for r in rows]
        assert len(ids) == len(set(ids)) == 4 and changed <= set(ids)
        assert set(ids) - changed <= unchanged
        assert all(r == by_id[r["review_id"]] for r in rows)
        assert all(r["meaning_preserved"] is None and r["reviewer"] is None and r["notes"] == ""
                   for r in rows)
        # Compatible with the frozen annotation validator.
        review._merge_annotations(source, tmp_path / "out" / name)
    assert meta["files"]["reviewer-a.jsonl"]["sha256"] == hashlib.sha256(
        (tmp_path / "out" / "reviewer-a.jsonl").read_bytes()).hexdigest()
    assert before == {p.name: p.read_bytes() for p in run.iterdir()}


def test_control_count_override_and_validation(run, tmp_path):
    meta = packet.build_packet(run, tmp_path / "zero", control_count=0)
    assert meta["counts"]["per_reviewer"] == 2
    with pytest.raises(ValueError, match="only 3 unchanged"):
        packet.build_packet(run, tmp_path / "many", control_count=4)
    with pytest.raises(ValueError, match="nonnegative"):
        packet.build_packet(run, tmp_path / "neg", control_count=-1)
    assert not (tmp_path / "many").exists() and not (tmp_path / "neg").exists()


def test_blind_content_has_no_arm_provider_or_savings(run, tmp_path):
    packet.build_packet(run, tmp_path / "out")
    keys = read_rows(run / "review-key.jsonl")
    text = "".join((tmp_path / "out" / n).read_text() for n in ("reviewer-a.jsonl", "reviewer-b.jsonl"))
    for row in read_rows(tmp_path / "out" / "reviewer-a.jsonl"):
        assert set(row) == {"review_id", "doc_id", "original", "suggested", "meaning_preserved",
                            "grammatical", "harmful_change", "reviewer", "notes"}
    for forbidden in ("provenance", "token", "selected", "config_hash", '"arm"', "provider", "saved",
                      '"b0"', '"b1"'):
        assert forbidden not in text.lower()
    assert not any(k["arm"] in ("b0", "b1") and f'"{k["arm"]}"' in text for k in keys)


def test_ordering_is_reproducible_and_reviewers_differ(frozen_run, tmp_path):
    a = packet.build_packet(frozen_run, tmp_path / "one")
    b = packet.build_packet(frozen_run, tmp_path / "two")
    assert a["files"] == b["files"]
    for name in ("reviewer-a.jsonl", "reviewer-b.jsonl"):
        assert (tmp_path / "one" / name).read_bytes() == (tmp_path / "two" / name).read_bytes()
    ids_a = [r["review_id"] for r in read_rows(tmp_path / "one" / "reviewer-a.jsonl")]
    ids_b = [r["review_id"] for r in read_rows(tmp_path / "one" / "reviewer-b.jsonl")]
    assert sorted(ids_a) == sorted(ids_b)
    assert a["files"]["reviewer-a.jsonl"]["sha256"] != a["files"]["reviewer-b.jsonl"]["sha256"] or ids_a != ids_b


def test_large_control_sample_is_capped_by_default(tmp_path):
    docs = [pilot.PilotDocument("c%d" % i, f"Plain prompt number {i}.", f"g{i}") for i in range(30)]
    docs += [pilot.PilotDocument(f"x{i}", f"They worked in order to finish. They prepared in order to improve number {i}.", f"h{i}") for i in range(25)]
    run = make_run(tmp_path / "big", docs)
    meta = packet.build_packet(run, tmp_path / "out")
    assert meta["counts"]["changed"] == 25 and meta["counts"]["controls"] == 20
    small = packet.build_packet(run, tmp_path / "small", control_count=3)
    assert small["counts"]["controls"] == 3


def test_no_changed_records_is_refused(tmp_path):
    run = make_run(tmp_path / "plain", [pilot.PilotDocument("p", "Plain text.", "g"),
                                        pilot.PilotDocument("q", "More plain text.", "h")])
    with pytest.raises(ValueError, match="no changed"):
        packet.build_packet(run, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_tampered_frozen_run_is_refused_without_output(run, tmp_path):
    with (run / "prompt-review.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="checksum mismatch: prompt-review.jsonl"):
        packet.build_packet(run, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_rehashed_inconsistent_export_is_refused(run, tmp_path):
    rows = read_rows(run / "prompt-review.jsonl")[:-1]
    (run / "prompt-review.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (run / "checksums.json").write_text(json.dumps({
        n: hashlib.sha256((run / n).read_bytes()).hexdigest() for n in review.RUN_FILES}))
    with pytest.raises(ValueError, match="exports do not match"):
        packet.build_packet(run, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_existing_output_is_never_overwritten(run, tmp_path):
    out = tmp_path / "out"
    packet.build_packet(run, out)
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    with pytest.raises(ValueError, match="new path"):
        packet.build_packet(run, out)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="new path"):
        packet.build_packet(run, empty)
    assert before == {p.name: p.read_bytes() for p in out.iterdir()}


def test_packet_output_cannot_contaminate_frozen_run(run):
    with pytest.raises(ValueError, match="inside the frozen run"):
        packet.build_packet(run, run / "packet")
    assert not (run / "packet").exists()


def test_cli_prints_metadata_and_reports_errors(run, tmp_path, capsys):
    out = tmp_path / "out"
    assert packet.main(["--run", str(run), "--out", str(out), "--control-count", "1"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["counts"]["controls"] == 1
    assert json.loads((out / "packet.json").read_text())["files"] == printed["files"]
    assert packet.main(["--run", str(run), "--out", str(out)]) == 2
    assert "new path" in capsys.readouterr().err
