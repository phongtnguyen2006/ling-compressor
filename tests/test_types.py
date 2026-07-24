import dataclasses
import pytest
from promptcomp.types import (
    BlockKind, Span, Block, Deletion, Veto, Substitution, CompressionResult,
)


def test_span_text_slices_source():
    span = Span(start=6, end=11)
    assert span.text("hello world") == "world"


def test_block_kind_values():
    assert BlockKind.PROSE.value == "prose"
    assert BlockKind.PASSTHROUGH.value == "passthrough"


def test_result_is_frozen():
    r = CompressionResult(
        original="a", compressed="a", tokens_before=1, tokens_after=1,
        token_counter="tiktoken:o200k_base", ratio=1.0, deleted=(), vetoed=(),
        substitutions={}, protect_list_version="none", config_hash="x",
        timings_ms={}, warnings=(),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.compressed = "b"


def test_block_frozen_and_offsets():
    b = Block(kind=BlockKind.PROSE, text="hi", start=0, end=2)
    assert b.text == "hi" and b.start == 0 and b.end == 2


def test_candidate_fields_and_frozen():
    from promptcomp.types import Candidate
    c = Candidate(
        deprel="prep", char_start=10, char_end=25,
        head_text="in", text="in the morning", n_tokens=3,
    )
    assert c.deprel == "prep"
    assert c.char_start == 10 and c.char_end == 25
    assert c.head_text == "in" and c.text == "in the morning"
    assert c.n_tokens == 3
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.deprel = "advmod"
