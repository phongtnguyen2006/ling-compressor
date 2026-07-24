import pytest

from promptcomp import compress
from promptcomp.parse import parse
from promptcomp.segment import segment
from promptcomp.types import BlockKind

DOCS = [
    "The committee approved the annual budget in the morning after a long debate. "
    "The vendor, acting in good faith, delivered the goods quickly and without delay.",
    "The Company shall reimburse expenses of $2,500 within thirty days, provided that "
    "receipts are submitted, and no exception applies unless approved in writing.",
]


@pytest.mark.parametrize("doc", DOCS)
@pytest.mark.parametrize("ratio", [0.3, 0.5, 0.7])
def test_every_output_sentence_has_a_root(doc, ratio):
    result = compress(doc, target_ratio=ratio, min_tokens=0)
    for block in segment(result.compressed):
        if block.kind is not BlockKind.PROSE:
            continue
        parsed = parse(block.text)
        for sent in parsed.sents:
            if sent.text.strip():
                assert any(t.dep_ == "ROOT" for t in sent), f"no ROOT in: {sent.text!r}"
