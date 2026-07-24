# tests/test_protect_invariants.py
from collections import Counter
from pathlib import Path

import pytest
from hypothesis import given, settings, strategies as st

from promptcomp import compress
from promptcomp.protect import extract_numbers, extract_negations
from promptcomp.segment import segment
from promptcomp.types import BlockKind

_SAMPLES = Path(__file__).resolve().parents[1] / "data" / "samples"
CORPUS = [p.read_text() for p in sorted(_SAMPLES.glob("*.txt"))]

# Curated sentences that stress each protected class.
CURATED = [
    "The Company shall reimburse expenses of $2,500 within thirty days after approval.",
    "Employees may not claim more than 15% above the standard rate in any month.",
    "No exception applies unless approved in writing by the regional director beforehand.",
    "Acme Corporation must deliver the goods no later than March 3, 2026, without fail.",
    "The vendor, acting in good faith, will provide at least two reports before the deadline.",
]


def _numbers_multiset(s):
    return Counter(extract_numbers(s))


@pytest.mark.parametrize("doc", CORPUS + CURATED)
@pytest.mark.parametrize("ratio", [0.3, 0.5, 0.7])
def test_numbers_multiset_preserved(doc, ratio):
    result = compress(doc, target_ratio=ratio, min_tokens=0)
    assert _numbers_multiset(result.original) == _numbers_multiset(result.compressed)


@pytest.mark.parametrize("doc", CORPUS + CURATED)
@pytest.mark.parametrize("ratio", [0.3, 0.5, 0.7])
def test_negation_count_non_decreasing(doc, ratio):
    result = compress(doc, target_ratio=ratio, min_tokens=0)
    before = len(extract_negations(result.original))
    after = len(extract_negations(result.compressed))
    assert after >= before


@pytest.mark.parametrize("doc", CORPUS + CURATED)
def test_no_vetoed_class_token_deleted(doc):
    # Every negation and number present originally must still be present.
    result = compress(doc, target_ratio=0.4, min_tokens=0)
    assert set(extract_numbers(result.original)) <= set(extract_numbers(result.compressed))
    assert Counter(extract_negations(result.compressed)) >= Counter(extract_negations(result.original))


@pytest.mark.parametrize("doc", CORPUS + CURATED)
def test_passthrough_blocks_byte_identical(doc):
    result = compress(doc, target_ratio=0.4, min_tokens=0)
    for block in segment(result.original):
        if block.kind is BlockKind.PASSTHROUGH:
            assert block.text in result.compressed


@settings(max_examples=40, deadline=None)
@given(
    template=st.sampled_from(CURATED),
    ratio=st.floats(min_value=0.2, max_value=0.9),
)
def test_property_numbers_and_negation_preserved(template, ratio):
    result = compress(template, target_ratio=ratio, min_tokens=0)
    assert _numbers_multiset(template) == _numbers_multiset(result.compressed)
    assert len(extract_negations(result.compressed)) >= len(extract_negations(template))
