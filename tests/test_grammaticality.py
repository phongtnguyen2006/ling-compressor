from pathlib import Path

import pytest

from promptcomp import compress
from promptcomp.parse import parse
from promptcomp.segment import segment
from promptcomp.types import BlockKind

_POLICY_SNIP = (
    Path(__file__).resolve().parent.parent / "data" / "samples" / "policy_snip.txt"
).read_text().strip()

DOCS = [
    # Confirmed empirically to delete tokens at ratios 0.3/0.5/0.7.
    "The committee approved the annual budget in the morning after a long debate. "
    "The vendor, acting in good faith, delivered the goods quickly and without delay.",
    "The auditor, who arrived early that rainy morning, confirmed the balance of "
    "$4,200 without objection, and the clerk filed the report quietly before noon.",
    _POLICY_SNIP,
]


@pytest.mark.parametrize("doc", DOCS)
@pytest.mark.parametrize("ratio", [0.3, 0.5, 0.7])
def test_every_output_sentence_has_a_root(doc, ratio):
    result = compress(doc, target_ratio=ratio, min_tokens=0)
    assert len(result.deleted) > 0, "fixture no longer exercises deletion; grammaticality guard is vacuous"
    for block in segment(result.compressed):
        if block.kind is not BlockKind.PROSE:
            continue
        parsed = parse(block.text)
        for sent in parsed.sents:
            if sent.text.strip():
                assert any(t.dep_ == "ROOT" for t in sent), f"no ROOT in: {sent.text!r}"
