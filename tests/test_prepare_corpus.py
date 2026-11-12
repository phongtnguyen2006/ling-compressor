"""Small selection fixtures verify the published corpus preparation contract."""
from __future__ import annotations

import json

import pytest

from eval.prepare_ag_news import LABELS, prepare as prepare_ag_news
from eval.prepare_dolly import CATEGORIES, prepare as prepare_dolly


def test_dolly_selector_freezes_balanced_splits_and_rejects_overwrite(tmp_path):
    source = tmp_path / "dolly.jsonl"
    rows = []
    for category in CATEGORIES:
        for i in range(4):
            rows.append({
                "category": category,
                "instruction": f"Explain {category} scenario number {i} in detail please",
                "context": f"Distinct context for {category} case {i} and its implications.",
                "response": f"Reference {category} {i}",
            })
    source.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    out = tmp_path / "selection"
    meta = prepare_dolly(source, out, per_category=1)
    prompts = [json.loads(line) for line in (out / "prompts.jsonl").read_text().splitlines()]
    assert meta["documents"] == 2 * len(CATEGORIES)
    assert {row["split"] for row in prompts} == {"development", "held-out"}
    assert all(row["group"].split(":", 1)[0] in CATEGORIES for row in prompts)
    assert len({row["text"] for row in prompts}) == len(prompts)
    with pytest.raises(ValueError, match="already exists"):
        prepare_dolly(source, out, per_category=1)


def test_ag_news_selector_keeps_category_in_group_and_labels_local(tmp_path):
    parquet = pytest.importorskip("pyarrow.parquet")
    pa = pytest.importorskip("pyarrow")
    paths = {}
    for split in ("train", "test"):
        rows = [{"text": f"{LABELS[label]} {split} article about a distinct item {label} "
                         + "relevant news details and background facts " * 4,
                 "label": label} for label in range(4)]
        path = tmp_path / f"{split}.parquet"
        parquet.write_table(pa.Table.from_pylist(rows), path)
        paths[split] = path
    out = tmp_path / "selection"
    meta = prepare_ag_news(paths["train"], paths["test"], out, per_label=1)
    prompts = [json.loads(line) for line in (out / "prompts.jsonl").read_text().splitlines()]
    assert meta["documents"] == 8
    assert {row["split"] for row in prompts} == {"development", "held-out"}
    assert all(row["group"].split(":", 1)[0] == row["category"] for row in prompts)
    assert all(row["category"] == LABELS[row["gold_label"]] for row in prompts)
