"""Threshold policy for vector-generated suggestions."""

import pytest

from promptcomp.lexical.engine import suggest
from promptcomp.lexical.models import LexicalCandidate


class CharacterCounter:
    name = "fixture:characters"

    def count(self, text):
        return len(text)


class VectorProvider:
    name = "fixture:vectors"
    max_phrase_words = 1

    def __init__(self, similarity):
        self.similarity = similarity

    def candidates(self, source, *, pos=None):
        if source == "large":
            return (LexicalCandidate(source, "vast", "embedding:fixture", self.similarity, "ADJ"),)
        return ()


def test_adaptive_threshold_rejects_marginal_one_token_gain():
    text = "The large plan works."
    static = suggest(text, VectorProvider(0.82), counter=CharacterCounter(),
                     min_embedding_similarity=0.8)
    adaptive = suggest(text, VectorProvider(0.82), counter=CharacterCounter(),
                       min_embedding_similarity=0.8, adaptive_similarity=True)
    assert static.suggested == "The vast plan works."
    assert adaptive.suggested == text
    assert adaptive.decisions[0].reason == "embedding_similarity_below_threshold"
    assert static.config_hash != adaptive.config_hash


def test_missing_score_is_rejected_when_threshold_is_requested():
    result = suggest("The large plan works.", VectorProvider(None),
                     counter=CharacterCounter(), min_embedding_similarity=0.8)
    assert result.edits == ()
    assert result.decisions[0].reason == "embedding_similarity_unavailable"


@pytest.mark.parametrize("threshold, adaptive", [(float("nan"), False), (1.1, False), (None, True)])
def test_invalid_threshold_configuration_is_rejected(threshold, adaptive):
    with pytest.raises(ValueError):
        suggest("The large plan works.", VectorProvider(0.9), counter=CharacterCounter(),
                min_embedding_similarity=threshold, adaptive_similarity=adaptive)
