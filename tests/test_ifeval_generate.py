"""T-29 generation runner: selection, provenance, validation (no real model)."""
import argparse
import hashlib
import json
from pathlib import Path

import pytest

from eval import ifeval_generate as gen

EVIDENCE = (Path(__file__).resolve().parents[1] / "docs" / "semantic-token-compression"
            / "evidence" / "2026-10-07-lexical-smoke")
REV = "7ae557604adf67be50417f59c2c2f167def9a775"


def _args(tmp_path, **kw):
    model = tmp_path / "model"
    model.mkdir(exist_ok=True)
    base = dict(run=EVIDENCE, model_path=model, model_id="Qwen/Qwen2.5-0.5B-Instruct",
                model_revision=REV, out=tmp_path / "out", doc_ids=None, limit=1, split=None,
                changed_arm=None, max_new_tokens=8)
    base.update(kw)
    return argparse.Namespace(**base)


def _fake_loader(path):
    return object(), object(), "cpu"


def _fake_generator(tok, model, device, prompt, max_new):
    return "resp:" + prompt[:5], 11, 3, len(prompt) % 2 == 0


def _results():
    return [json.loads(x) for x in (EVIDENCE / "results.jsonl").read_text().splitlines() if x]


def test_run_writes_exact_fields_and_provenance(tmp_path):
    args = _args(tmp_path, limit=2, changed_arm="b2")
    m = gen.run(args, _fake_loader, _fake_generator)
    rows = [json.loads(x) for x in (args.out / "responses.jsonl").read_text().splitlines()]
    assert len(rows) == 4 and m["arms"] == ["u", "b2"]
    src = {(r["doc_id"], r["arm"]): r for r in _results()}
    for row in rows:
        assert tuple(sorted(row)) == tuple(sorted(gen.FIELDS))
        prompt = src[(row["doc_id"], row["arm"])]["suggested"]
        assert row["prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()
        assert row["decoding"] == {"do_sample": False, "max_new_tokens": 8}
        assert row["model_revision"] == REV and row["model_input_tokens"] == 11
        assert row["output_tokens"] == 3
    assert rows == sorted(rows, key=lambda r: (r["doc_id"], ["u", "b2"].index(r["arm"])))
    saved = json.loads((args.out / "manifest.json").read_text())
    assert saved["source_run_fingerprint"] == json.loads((EVIDENCE / "manifest.json").read_text())["run_fingerprint"]
    assert saved["selected_doc_ids"] == m["selected_doc_ids"] and len(saved["truncation"]) == 4
    assert saved["files"]["responses.jsonl"] == hashlib.sha256((args.out / "responses.jsonl").read_bytes()).hexdigest()
    assert saved["model"]["local_files_only"] is True and "transformers" in saved["versions"]
    assert "no quality claim" in saved["note"]


def test_deterministic_selection_and_doc_id_file(tmp_path):
    ids = sorted({r["doc_id"] for r in _results()})
    f = tmp_path / "ids.txt"
    f.write_text(ids[-1] + "\n")
    m = gen.run(_args(tmp_path, doc_ids=f, limit=None), _fake_loader, _fake_generator)
    assert m["selected_doc_ids"] == [ids[-1]]
    j = tmp_path / "ids.jsonl"
    j.write_text(json.dumps({"doc_id": ids[0]}) + "\n" + json.dumps(ids[1]) + "\n")
    assert gen.read_doc_ids(j) == [ids[0], ids[1]]


def test_validation_errors(tmp_path):
    dup = tmp_path / "dup.txt"
    dup.write_text("a\na\n")
    with pytest.raises(ValueError, match="duplicate"):
        gen.read_doc_ids(dup)
    unknown = tmp_path / "unk.txt"
    unknown.write_text("nope\n")
    with pytest.raises(ValueError, match="not available"):
        gen.run(_args(tmp_path, doc_ids=unknown, limit=None), _fake_loader, _fake_generator)
    with pytest.raises(ValueError, match="changed arm"):
        gen.run(_args(tmp_path, changed_arm="zz"), _fake_loader, _fake_generator)
    with pytest.raises(ValueError, match="exactly one"):
        gen.run(_args(tmp_path, limit=None), _fake_loader, _fake_generator)
    with pytest.raises(ValueError, match="no documents"):
        gen.run(_args(tmp_path, split="held-out"), _fake_loader, _fake_generator)


def test_refuses_overwrite_and_bad_revision(tmp_path):
    args = _args(tmp_path)
    args.out.mkdir()
    with pytest.raises(FileExistsError):
        gen.run(args, _fake_loader, _fake_generator)
    snap = tmp_path / REV
    snap.mkdir()
    with pytest.raises(ValueError, match="revision"):
        gen.run(_args(tmp_path, model_path=snap, model_revision="0" * 40, out=tmp_path / "o2"),
                _fake_loader, _fake_generator)
    assert not (tmp_path / "o2").exists()


def test_cli_returns_error_code_without_loading_model(tmp_path, capsys):
    rc = gen.main(["--run", str(EVIDENCE), "--model-path", str(tmp_path / "missing"),
                   "--model-id", "x", "--model-revision", "y", "--out", str(tmp_path / "o"),
                   "--limit", "1"])
    assert rc == 2 and "not a local directory" in capsys.readouterr().err
