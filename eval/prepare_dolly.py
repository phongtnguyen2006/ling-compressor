"""Freeze local Dolly prompt strata before running lexical candidate discovery.

Source: https://huggingface.co/datasets/databricks/databricks-dolly-15k
License on the dataset card: CC BY-SA 3.0. This selector writes source text
only to a caller chosen path; keep third-party prompts out of tracked code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

CATEGORIES = (
    "brainstorming", "classification", "closed_qa", "creative_writing",
    "general_qa", "information_extraction", "open_qa", "summarization",
)


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _words(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+", text.casefold()))


def _near_duplicate(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    if left == right or (len(left) >= 10 and len(right) >= 10 and left[:10] == right[:10]):
        return True
    a, b = set(zip(left, left[1:], left[2:])), set(zip(right, right[1:], right[2:]))
    return bool(a and b) and len(a & b) / len(a | b) >= 0.85


def prepare(source: Path, out: Path, *, per_category: int = 25) -> dict:
    if per_category < 1:
        raise ValueError("per_category must be positive")
    if out.exists():
        raise ValueError("output directory already exists; frozen selections are not overwritten")
    by_category: dict[str, list[tuple[str, str, str, tuple[str, ...]]]] = {
        name: [] for name in CATEGORIES
    }
    raw = source.read_bytes()
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("category") not in by_category:
            raise ValueError(f"unexpected category on line {line_number}")
        instruction, context, response = (row.get(field) for field in ("instruction", "context", "response"))
        if not all(isinstance(value, str) for value in (instruction, context, response)):
            raise ValueError(f"invalid text on line {line_number}")
        text = instruction.strip() + ("\n\nContext: " + context.strip() if context.strip() else "")
        words = _words(text)
        if not 5 <= len(words) <= 400:
            continue
        digest = _hash(text.encode("utf-8"))
        by_category[row["category"]].append((digest, text, response, words))

    prompts: list[dict] = []
    gold: list[dict] = []
    seen: list[tuple[str, ...]] = []
    selected_hashes: set[str] = set()
    # Freeze every development category before selecting any held-out prompt.
    # Held-out records may depend on the development selection, never vice versa.
    for split in ("development", "held-out"):
        for category in CATEGORIES:
            entries = sorted(by_category[category])
            taken = 0
            for digest, text, response, words in entries:
                if digest in selected_hashes or any(_near_duplicate(words, prior) for prior in seen):
                    continue
                selected_hashes.add(digest)
                seen.append(words)
                doc_id = f"dolly-{category}-{digest[:16]}"
                prompts.append({
                    "id": doc_id, "text": text, "group": f"{category}:{digest[:20]}",
                    "split": split,
                })
                gold.append({"id": doc_id, "response": response, "category": category, "split": split})
                taken += 1
                if taken == per_category:
                    break
            if taken != per_category:
                raise ValueError(f"not enough distinct prompts in {category}/{split}: {taken}")

    out.mkdir(parents=True)
    prompts_path = out / "prompts.jsonl"
    gold_path = out / "reference-responses.jsonl"
    for path, rows in ((prompts_path, prompts), (gold_path, gold)):
        with path.open("w", encoding="utf-8") as stream:
            for row in sorted(rows, key=lambda row: row["id"]):
                stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    metadata = {
        "schema_version": "dolly-pilot-v1",
        "source": "https://huggingface.co/datasets/databricks/databricks-dolly-15k",
        "source_license": "CC BY-SA 3.0 per dataset card",
        "source_sha256": _hash(raw),
        "selection": "sorted SHA-256 of instruction plus context per category; choose all development strata before held-out; reject matching first 10 words or 3-gram Jaccard >= 0.85 among selected prompts",
        "categories": CATEGORIES,
        "per_category_per_split": per_category,
        "documents": len(prompts),
        "prompt_sha256": _hash(prompts_path.read_bytes()),
        "reference_sha256": _hash(gold_path.read_bytes()),
        "scope": "Dolly instruction/context proxy, not actual production customer workload",
    }
    (out / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-category", type=int, default=25)
    args = parser.parse_args(argv)
    print(json.dumps(prepare(args.source, args.out, per_category=args.per_category), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
