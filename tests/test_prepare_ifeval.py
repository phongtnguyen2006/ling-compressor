import json

import pytest

from eval import prepare_ifeval as pi
from eval.lexical_pilot import load_documents


def rec(i, **kw):
    r = {"key": i, "prompt": f"Write essay number {i}.",
         "instruction_id_list": ["length_constraints:number_words"],
         "kwargs": [{"relation": "at least", "num_words": 100 + i, "x": None}]}
    r.update(kw)
    return r


def write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


@pytest.fixture
def src(tmp_path):
    return write(tmp_path / "src.jsonl", [rec(i) for i in range(1, 201)])


def test_freeze_valid_and_pilot_compatible(tmp_path, src):
    out = tmp_path / "out"
    prov = pi.prepare(src, out, expected_count=200)
    docs = load_documents(out / "prompts.jsonl")
    assert len(docs) == 200 and {d.id for d in docs} == {f"ifeval-{i}" for i in range(1, 201)}
    gold = [json.loads(x) for x in (out / "gold.jsonl").read_text().splitlines()]
    by = {g["id"]: g for g in gold}
    assert by["ifeval-7"]["prompt"] == rec(7)["prompt"]
    assert by["ifeval-7"]["kwargs"] == rec(7)["kwargs"]
    assert by["ifeval-7"]["source_sha256"] == prov["source_sha256"]
    assert prov["records"] == 200
    for name in ("prompts.jsonl", "gold.jsonl"):
        assert pi._sha256((out / name).read_bytes()) == prov["file_sha256"][name]
    assert json.loads((out / "provenance.json").read_text()) == prov
    assert "Apache" in prov["license_source"]


def test_split_coverage_and_ratio(tmp_path, src):
    prov = pi.prepare(src, tmp_path / "o", expected_count=200)
    c = prov["split_counts"]
    assert c["development"] + c["held-out"] == 200
    assert 20 <= c["development"] <= 60 and c["held-out"] > 0
    assert pi.split_for_key(5) == pi.split_for_key(5)


def test_reproducible(tmp_path, src):
    pi.prepare(src, tmp_path / "a", expected_count=200)
    pi.prepare(src, tmp_path / "b", expected_count=200)
    for n in ("prompts.jsonl", "gold.jsonl", "provenance.json"):
        assert (tmp_path / "a" / n).read_bytes() == (tmp_path / "b" / n).read_bytes()


def test_order_independent(tmp_path, src):
    rows = [rec(i) for i in range(200, 0, -1)]
    pi.prepare(write(tmp_path / "rev.jsonl", rows), tmp_path / "r", expected_count=200)
    pi.prepare(src, tmp_path / "f", expected_count=200)
    assert (tmp_path / "r/prompts.jsonl").read_bytes() == (tmp_path / "f/prompts.jsonl").read_bytes()


def test_reject_overwrite(tmp_path, src):
    (tmp_path / "exists").mkdir()
    with pytest.raises(FileExistsError):
        pi.prepare(src, tmp_path / "exists", expected_count=200)
    assert pi.main(["--source", str(src), "--out", str(tmp_path / "exists"),
                    "--expected-count", "200"]) == 1


def bad_cases():
    return {
        "dup_key": [rec(1), rec(1, prompt="other")],
        "dup_prompt": [rec(1), rec(2, prompt=rec(1)["prompt"])],
        "empty_prompt": [rec(1, prompt="  ")],
        "nonstring_prompt": [rec(1, prompt=3)],
        "bool_key": [rec(1, key=True)],
        "str_key": [rec(1, key="1")],
        "empty_ids": [rec(1, instruction_id_list=[], kwargs=[])],
        "blank_id": [rec(1, instruction_id_list=[""])],
        "len_mismatch": [rec(1, kwargs=[{}, {}])],
        "kwargs_not_dict": [rec(1, kwargs=["x"])],
        "extra_field": [dict(rec(1), extra=1)],
        "missing_field": [{k: v for k, v in rec(1).items() if k != "kwargs"}],
        "no_records": [],
    }


@pytest.mark.parametrize("name", list(bad_cases()))
def test_schema_rejects(tmp_path, name):
    s = write(tmp_path / "s.jsonl", bad_cases()[name])
    with pytest.raises(ValueError):
        pi.prepare(s, tmp_path / "out", expected_count=None)
    assert not (tmp_path / "out").exists()


def test_rejects_bad_json_and_count(tmp_path, src):
    s = tmp_path / "s.jsonl"
    s.write_text("{not json\n")
    with pytest.raises(ValueError):
        pi.prepare(s, tmp_path / "o1", expected_count=None)
    with pytest.raises(ValueError, match="expected 541"):
        pi.prepare(src, tmp_path / "o2")  # default requires full 541


def test_tamper_changes_hashes(tmp_path, src):
    p1 = pi.prepare(src, tmp_path / "a", expected_count=200)
    rows = [rec(i) for i in range(1, 201)]
    rows[3]["kwargs"][0]["num_words"] += 1
    p2 = pi.prepare(write(tmp_path / "t.jsonl", rows), tmp_path / "b", expected_count=200)
    assert p1["source_sha256"] != p2["source_sha256"]
    assert p1["file_sha256"]["gold.jsonl"] != p2["file_sha256"]["gold.jsonl"]
