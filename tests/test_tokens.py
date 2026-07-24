import pytest
from promptcomp.tokens import TiktokenCounter, CachedCounter, AnthropicCounter


def test_tiktoken_counts_are_positive_and_named():
    c = TiktokenCounter()
    assert c.name == "tiktoken:o200k_base"
    assert c.count("hello world, this is a test") > 3


def test_tiktoken_empty_is_zero():
    assert TiktokenCounter().count("") == 0


class _CountingCounter:
    """Counts how many times the inner counter is actually invoked."""
    name = "tiktoken:o200k_base"

    def __init__(self):
        self.calls = 0

    def count(self, text: str) -> int:
        self.calls += 1
        return len(text.split())


def test_cache_avoids_recount(tmp_path):
    inner = _CountingCounter()
    cached = CachedCounter(inner, db_path=tmp_path / "cache.sqlite")
    assert cached.count("a b c") == 3
    assert cached.count("a b c") == 3
    assert inner.calls == 1  # second call served from cache


def test_cache_persists_across_instances(tmp_path):
    db = tmp_path / "cache.sqlite"
    inner1 = _CountingCounter()
    CachedCounter(inner1, db_path=db).count("x y")
    inner2 = _CountingCounter()
    assert CachedCounter(inner2, db_path=db).count("x y") == 2
    assert inner2.calls == 0  # served from disk cache written by inner1


def test_anthropic_counter_missing_dep_is_actionable():
    # Without the anthropic package / key, constructing-then-counting must raise
    # a clear RuntimeError rather than an opaque ImportError.
    c = AnthropicCounter()
    with pytest.raises(RuntimeError) as exc:
        c.count("hello")
    msg = str(exc.value).lower()
    assert "anthropic" in msg
