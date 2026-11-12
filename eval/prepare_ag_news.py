"""Freeze a balanced, local AG News pilot manifest from user-provisioned Parquet files.

The dataset card lists its license as unknown. Keep both Parquet and generated
article text local; do not publish the manifest or treat it as a production
customer workload. The fixed classification instruction is outside each text.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

LABELS = ("World", "Sports", "Business", "Sci/Tech")
SCHEMA = "ag-news-pilot-v1"


def _sha(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _words(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+", text.casefold()))


def _near_duplicate(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    if not left or not right:
        return False
    if left[:10] == right[:10]:
        return True
    a, b = set(zip(left, left[1:], left[2:])), set(zip(right, right[1:], right[2:]))
    return bool(a and b) and len(a & b) / len(a | b) >= 0.8


def prepare(train: Path, test: Path, out: Path, *, per_label: int = 50) -> dict:
    if per_label < 1:
        raise ValueError("per_label must be positive")
    if out.exists():
        raise ValueError("output directory already exists; frozen selections are not overwritten")
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("pyarrow is needed to read local AG News Parquet files") from exc
    selected: list[dict] = []
    seen: list[tuple[str, ...]] = []
    for split, source in (("development", train), ("held-out", test)):
        table = parquet.read_table(source, columns=["text", "label"])
        if table.column_names != ["text", "label"]:
            raise ValueError(f"unexpected AG News schema: {source}")
        by_label: dict[int, list[tuple[str, str, tuple[str, ...]]]] = {i: [] for i in range(4)}
        for record in table.to_pylist():
            text, label = record["text"], record["label"]
            if not isinstance(text, str) or type(label) is not int or label not in by_label:
                raise ValueError(f"invalid AG News row: {source}")
            text = text.strip()
            words = _words(text)
            if len(words) < 20 or len(words) > 300:
                continue
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            by_label[label].append((digest, text, words))
        for label in range(4):
            count = 0
            for digest, text, words in sorted(by_label[label]):
                if any(_near_duplicate(words, prior) for prior in seen):
                    continue
                seen.append(words)
                selected.append({
                    "id": f"ag-{split}-{label}-{count:03d}",
                    "text": text,
                    "group": f"{LABELS[label]}:article-{digest[:20]}",
                    "split": split,
                    "category": LABELS[label],
                    "gold_label": label,
                    "source_sha256": digest,
                })
                count += 1
                if count == per_label:
                    break
            if count != per_label:
                raise ValueError(f"insufficient distinct AG News articles for {split}/{LABELS[label]}")
    out.mkdir(parents=True)
    manifest = out / "prompts.jsonl"
    with manifest.open("w", encoding="utf-8") as stream:
        for row in sorted(selected, key=lambda row: row["id"]):
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    metadata = {
        "schema_version": SCHEMA,
        "source": "https://huggingface.co/datasets/fancyzhx/ag_news",
        "source_license": "unknown on dataset card; local evaluation only; do not redistribute article text",
        "source_files": {"train": _sha(train), "test": _sha(test)},
        "selection": "sorted SHA-256 of article text, per label and official source split; 20-300 word articles; reject matching first 10 words or 3-gram Jaccard >= 0.80 across selected records",
        "labels": LABELS,
        "per_label_per_split": per_label,
        "documents": len(selected),
        "prompt_sha256": _sha(manifest),
        "fixed_instruction_outside_text": "Classify the article as World, Sports, Business, or Sci/Tech.",
        "scope": "AG News topic classification proxy, not representative production customer prompts",
    }
    (out / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-label", type=int, default=50)
    args = parser.parse_args(argv)
    print(json.dumps(prepare(args.train, args.test, args.out, per_label=args.per_label), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
