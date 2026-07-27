from eval.baselines import (
    NoneCompressor, StopwordCompressor, OursCompressor, available_baselines, Compressed,
)


def test_none_is_identity():
    c = NoneCompressor()
    out = c.compress("The committee approved the budget of $2,500 today.", 0.5)
    assert isinstance(out, Compressed)
    assert out.text == "The committee approved the budget of $2,500 today."
    assert c.name == "none"


def test_stopword_removes_common_words_and_is_floor():
    c = StopwordCompressor()
    out = c.compress("The vendor delivered the goods in the morning.", 0.5)
    assert c.name == "stopword"
    # naive stopword removal drops "the"/"in" — visibly lossy
    assert "the" not in out.text.lower().split()
    assert len(out.text) < len("The vendor delivered the goods in the morning.")


def test_ours_wraps_compress_and_deletes():
    c = OursCompressor()
    text = ("The committee approved the annual budget in the morning after a long debate, "
            "and the vendor delivered the goods quickly without any delay.")
    out = c.compress(text, 0.5)
    assert c.name == "ours"
    assert len(out.text) <= len(text)


def test_available_baselines_includes_offline_three():
    names = {c.name for c in available_baselines()}
    assert {"none", "stopword", "ours"} <= names
