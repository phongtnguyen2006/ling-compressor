from hypothesis import given, strategies as st
from promptcomp.segment import segment
from promptcomp.types import BlockKind


def _reassembles(text: str) -> bool:
    blocks = segment(text)
    if not blocks:
        return text == ""
    if blocks[0].start != 0 or blocks[-1].end != len(text):
        return False
    for a, b in zip(blocks, blocks[1:]):
        if a.end != b.start:
            return False
    return "".join(b.text for b in blocks) == text


def test_reassembly_exact_simple():
    assert _reassembles("The quick brown fox.\nAnother sentence here.\n")


def test_fenced_code_is_passthrough():
    text = "Intro prose line.\n```\ncode = 1\n```\nOutro prose.\n"
    blocks = segment(text)
    kinds = [(b.kind, b.text) for b in blocks]
    assert any(k is BlockKind.PASSTHROUGH and "code = 1" in t for k, t in kinds)
    assert any(k is BlockKind.PROSE and "Intro prose" in t for k, t in kinds)
    assert _reassembles(text)


def test_digit_heavy_line_is_passthrough():
    text = "Normal explanatory sentence about the account.\n1234 5678 9012 3456\n"
    blocks = segment(text)
    assert any(
        b.kind is BlockKind.PASSTHROUGH and "1234 5678" in b.text for b in blocks
    )


def test_url_line_is_passthrough():
    text = "See the reference below for details.\nhttps://example.com/a/b/c\n"
    blocks = segment(text)
    assert any(
        b.kind is BlockKind.PASSTHROUGH and "example.com" in b.text for b in blocks
    )


def test_markdown_table_is_passthrough():
    text = (
        "Here is the breakdown.\n"
        "| Col A | Col B |\n"
        "| --- | --- |\n"
        "| 1 | 2 |\n"
        "Closing remark.\n"
    )
    blocks = segment(text)
    assert any(b.kind is BlockKind.PASSTHROUGH and "Col A" in b.text for b in blocks)


def test_multi_row_table_stays_passthrough():
    text = (
        "Here is the breakdown.\n"
        "| Name | Role |\n"
        "| --- | --- |\n"
        "| Alice | Engineer |\n"
        "| Bob | Manager |\n"
        "Closing remark.\n"
    )
    blocks = segment(text)
    assert any(
        b.kind is BlockKind.PASSTHROUGH and "Alice" in b.text and "Engineer" in b.text
        for b in blocks
    )
    assert any(
        b.kind is BlockKind.PASSTHROUGH and "Bob" in b.text and "Manager" in b.text
        for b in blocks
    )
    assert any(
        b.kind is BlockKind.PROSE and "Closing remark" in b.text for b in blocks
    )
    assert "".join(b.text for b in blocks) == text


def test_indented_at_prose_is_not_stack_trace():
    text = "Some intro.\n  at the crossroads she paused and waited.\n"
    blocks = segment(text)
    target = next(
        b for b in blocks if "at the crossroads she paused" in b.text
    )
    assert target.kind is BlockKind.PROSE


@given(st.text())
def test_reassembly_never_loses_bytes(text):
    assert _reassembles(text)
