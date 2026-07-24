from eval.metrics import protected_violation_count, compression_metrics
from promptcomp.tokens import TiktokenCounter


def test_ours_has_zero_violations():
    # ours never deletes a protected token
    from eval.baselines import OursCompressor
    text = ("The company shall reimburse expenses of $2,500 within thirty days, and no "
            "exception applies unless approved; the vendor delivered the goods in the morning.")
    compressed = OursCompressor().compress(text, 0.4).text
    assert protected_violation_count(text, compressed) == 0


def test_stopword_loses_negation_so_has_violations():
    text = "The vendor did not deliver the goods and no exception applies."
    from eval.baselines import StopwordCompressor
    compressed = StopwordCompressor().compress(text, 0.4).text
    # stopword removal drops "not"/"no" -> protected negation lost
    assert protected_violation_count(text, compressed) > 0


def test_compression_metrics_shapes():
    m = compression_metrics("a " * 200, "a " * 100, TiktokenCounter())
    assert m["tokens_after"] < m["tokens_before"]
    assert 0 < m["compression_ratio"] <= 1.0
    assert m["token_counter"].startswith("tiktoken:")
