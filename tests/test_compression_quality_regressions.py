"""Prompt-level quality regressions from the IFEval and Dolly compression runs."""

import pytest

from promptcomp import compress
from promptcomp.lexical import PilotProvider, VectorIndex, suggest
from promptcomp.lexical.tokenization import local_counter


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="Tree deletion drops the required quotation format in IFEval prompt 163",
)
def test_tree_preserves_double_quote_instruction():
    prompt = (
        "Create a 4 day itinerary for a trip to Verona, Italy. Wrap your entire "
        'response with double quotes and include the letter "r" at least 12 times.'
    )
    result = compress(
        prompt, target_ratio=0.9, substitute=False, min_tokens=0,
        counter=local_counter("o200k_base"),
    )

    assert "Wrap your entire response with double quotes" in result.compressed


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="B2 selects a token-saving embedding neighbor that reverses the claim",
)
def test_embedding_preserves_polarity_of_undermines(tmp_path):
    # Sentence from dolly-closed_qa-02c700b5c297e0bf in the 400k GloVe run.
    sentence = (
        "Despite this evidence, that SPD researchers have yet to agree on a proven, "
        "standardized diagnostic tool undermines researchers' ability to define the "
        "boundaries of the disorder and makes correlational studies, like those on "
        "structural brain abnormalities, less convincing."
    )
    # Two synthetic vectors reproduce the observed neighbor at cosine ~0.7845.
    path = tmp_path / "vectors.txt"
    path.write_text("undermines 1 0\nreinforces 0.7845 0.6201\n", encoding="utf-8")
    provider = PilotProvider("b2", vectors=VectorIndex.from_text(path), phrases={})
    result = suggest(sentence, provider, counter=local_counter("o200k_base"))

    assert "undermines" in result.suggested and "reinforces" not in result.suggested
