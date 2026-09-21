from eval.baselines import Compressed
from eval.hybrid import SentinelHybridCompressor, mask_protected_clauses
from eval.metrics import protected_violation_count
from promptcomp.parse import parse


class FakeLLMLingua:
    def __init__(self, *, drop_sentinel: bool = False):
        self.drop_sentinel = drop_sentinel
        self.clean_calls = 0
        self.forced_calls = 0

    def compress(self, text: str, ratio: float) -> Compressed:
        self.clean_calls += 1
        return Compressed(text=text.replace("carefully ", ""), latency_ms=0.0)

    def compress_forced(self, text: str, ratio: float, force_tokens) -> Compressed:
        self.forced_calls += 1
        output = text.replace("carefully, ", "")
        if self.drop_sentinel:
            output = output.replace(force_tokens[0], "")
        return Compressed(text=output, latency_ms=0.0)


def test_masks_isolatable_protected_subordinate_clause():
    text = "The team worked carefully, unless spending exceeds $5,000."
    doc = parse(text)
    masked = mask_protected_clauses(doc, next(doc.sents))

    assert masked is not None
    assert masked.sentinels == ("KEEPCLAUSEA",)
    assert masked.clauses == ("unless spending exceeds $5,000",)
    assert masked.anchors == ("team", "worked")
    assert masked.marker_order == ("team", "worked", "KEEPCLAUSEA")
    assert masked.text == "The team worked carefully, KEEPCLAUSEA."


def test_sentinel_hybrid_restores_clause_byte_identically():
    text = "The team worked carefully, unless spending exceeds $5,000."
    backend = FakeLLMLingua()
    compressor = SentinelHybridCompressor(ll_compressor=backend)

    output = compressor.compress(text, 0.5)

    assert output.text == "The team worked unless spending exceeds $5,000."
    assert protected_violation_count(text, output.text) == 0
    assert backend.forced_calls == 1
    assert compressor.stats["masked_sents"] == 1
    assert compressor.stats["protected_clauses"] == 1


def test_protected_main_clause_falls_back_without_calling_llmlingua():
    text = "The company must pay $5,000 tomorrow."
    backend = FakeLLMLingua()
    compressor = SentinelHybridCompressor(ll_compressor=backend)

    output = compressor.compress(text, 0.5)

    assert protected_violation_count(text, output.text) == 0
    assert backend.clean_calls == 0
    assert backend.forced_calls == 0
    assert compressor.stats["fallback_sents"] == 1


def test_missing_sentinel_fails_closed_to_conservative_compressor():
    text = "The team worked carefully, unless spending exceeds $5,000."
    backend = FakeLLMLingua(drop_sentinel=True)
    compressor = SentinelHybridCompressor(ll_compressor=backend)

    output = compressor.compress(text, 0.5)

    assert protected_violation_count(text, output.text) == 0
    assert "unless spending exceeds $5,000" in output.text
    assert compressor.stats["validation_fallbacks"] == 1
