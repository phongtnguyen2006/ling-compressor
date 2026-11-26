"""Freeze a local official google/IFEval ``ifeval_input_data.jsonl`` (T-27).

No network. Writes prompts.jsonl (for ``eval.lexical_pilot``), gold.jsonl and
provenance.json into a NEW directory. Gold records are copied unaltered.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

SCHEMA_VERSION = "ifeval-frozen-v1"
EXPECTED_COUNT = 541
DEV_PERCENT = 20
LICENSE_LABEL = "google/IFEval (Apache-2.0), https://huggingface.co/datasets/google/IFEval"
FIELDS = ("key", "prompt", "instruction_id_list", "kwargs")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canon(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def _line(record) -> str:
    return json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"


def split_for_key(key: int) -> str:
    bucket = int(_sha256(f"ifeval-split-v1:{key}".encode())[:8], 16) % 100
    return "development" if bucket < DEV_PERCENT else "held-out"


def _no_nonfinite(value) -> None:
    if isinstance(value, float) and value != value or value in (float("inf"), float("-inf")):
        raise ValueError("non-finite number")
    if isinstance(value, dict):
        for k, v in value.items():
            if not isinstance(k, str):
                raise ValueError("kwargs keys must be strings")
            _no_nonfinite(v)
    elif isinstance(value, list):
        for v in value:
            _no_nonfinite(v)


def load_source(path: Path, expected_count: int | None = EXPECTED_COUNT) -> tuple[list[dict], bytes]:
    if not path.is_file():
        raise ValueError(f"source is not a file: {path}")
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"source is not UTF-8: {exc}") from exc
    records, keys, prompts = [], set(), set()
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("expected an object")
            if set(row) != set(FIELDS):
                raise ValueError(f"fields must be exactly {list(FIELDS)}")
            key, prompt, ids, kwargs = (row[f] for f in FIELDS)
            if type(key) is not int or key < 0:
                raise ValueError("key must be a non-negative int")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("prompt must be a nonempty string")
            if (not isinstance(ids, list) or not ids
                    or not all(isinstance(i, str) and i.strip() for i in ids)):
                raise ValueError("instruction_id_list must be a nonempty list of nonempty strings")
            if not isinstance(kwargs, list) or len(kwargs) != len(ids):
                raise ValueError("kwargs must be a list matching instruction_id_list length")
            if not all(isinstance(k, dict) for k in kwargs):
                raise ValueError("each kwargs entry must be an object")
            _no_nonfinite(kwargs)
            if key in keys:
                raise ValueError(f"duplicate key {key}")
            if prompt in prompts:
                raise ValueError("duplicate prompt")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"invalid IFEval record on line {n}: {exc}") from exc
        keys.add(key)
        prompts.add(prompt)
        records.append(row)
    if not records:
        raise ValueError("source contains no records")
    if expected_count is not None and len(records) != expected_count:
        raise ValueError(f"expected {expected_count} records, found {len(records)}")
    return sorted(records, key=lambda r: r["key"]), raw


def prepare(source: Path, out: Path, expected_count: int | None = EXPECTED_COUNT) -> dict:
    records, raw = load_source(source, expected_count)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {out}")
    source_sha = _sha256(raw)
    prompts, gold, counts = [], [], {"development": 0, "held-out": 0}
    for r in records:
        doc_id = f"ifeval-{r['key']}"
        split = split_for_key(r["key"])
        counts[split] += 1
        prompts.append({"id": doc_id, "text": r["prompt"], "group": doc_id, "split": split})
        gold.append({"id": doc_id, "key": r["key"], "prompt": r["prompt"],
                     "instruction_id_list": r["instruction_id_list"], "kwargs": r["kwargs"],
                     "source_sha256": source_sha, "record_sha256": _sha256(_canon(r).encode())})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir()  # raises if it appeared meanwhile
    files = {"prompts.jsonl": prompts, "gold.jsonl": gold}
    hashes = {}
    for name, rows in files.items():
        data = "".join(_line(x) for x in rows).encode("utf-8")
        (out / name).write_bytes(data)
        hashes[name] = _sha256(data)
    prov = {"schema_version": SCHEMA_VERSION, "source_file": source.name,
            "source_sha256": source_sha, "license_source": LICENSE_LABEL,
            "records": len(records), "split_counts": counts,
            "split_rule": f"sha256('ifeval-split-v1:'+key)[:8] % 100 < {DEV_PERCENT} -> development",
            "file_sha256": hashes}
    (out / "provenance.json").write_text(json.dumps(prov, indent=2, sort_keys=True) + "\n",
                                         encoding="utf-8")
    return prov


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Freeze local official IFEval input data.")
    p.add_argument("--source", type=Path, required=True, help="local ifeval_input_data.jsonl")
    p.add_argument("--out", type=Path, required=True, help="NEW output directory")
    p.add_argument("--expected-count", type=int, default=EXPECTED_COUNT,
                   help="required record count (default 541, the full official set)")
    a = p.parse_args(argv)
    try:
        prov = prepare(a.source, a.out, a.expected_count)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"out": str(a.out), "records": prov["records"],
                      "split_counts": prov["split_counts"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
