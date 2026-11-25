"""Synthetic vectors exercise mechanics only; they are not semantic evidence."""
import json
import sys
from dataclasses import FrozenInstanceError

import pytest

from promptcomp.lexical.candidates import PilotProvider, VectorIndex
from promptcomp.lexical.models import LexicalCandidate


@pytest.fixture
def synthetic_vectors(tmp_path):
    path = tmp_path / "synthetic-vectors.txt"
    path.write_text(
        "6 3\nsource 2 0 0\nzeta 2 0 0\nalpha 0 1 0\n"
        "beta 2 0 0\ngamma 1 1 0\nother -1 0 0\n"
    )
    return VectorIndex.from_text(path)


class FakeLemma:
    def __init__(self, name):
        self._name = name

    def name(self):
        return self._name


class FakeSynset:
    def name(self):
        return "source.n.01"

    def pos(self):
        return "n"

    def lemmas(self):
        return [FakeLemma(name) for name in ("source", "zeta", "alpha", "beta", "missing")]


class FakeWordNet:
    def synsets(self, source, pos=None):
        return [FakeSynset()] if source == "source" and pos in (None, "n") else []


@pytest.fixture
def fake_wordnet(monkeypatch):
    monkeypatch.setattr(
        "promptcomp.lexical.candidates._load_wordnet",
        lambda: (FakeWordNet(), {"status": "synthetic-test-fixture", "version": "fake", "sha256": "fake"}),
    )


def identities(candidates):
    return {(c.replacement, c.provenance, c.pos) for c in candidates}


def test_candidate_contract_is_frozen():
    candidate = LexicalCandidate("long", "short", "curated")
    with pytest.raises(FrozenInstanceError):
        candidate.replacement = "other"


def test_vector_cosine_neighbors_are_normalized_and_ties_are_lexical(synthetic_vectors):
    assert synthetic_vectors.similarity("source", "zeta") == pytest.approx(1)
    assert synthetic_vectors.similarity("source", "other") == pytest.approx(-1)
    assert synthetic_vectors.similarity("source", "gamma") == pytest.approx(2 ** -0.5)
    assert synthetic_vectors.nearest("source", k=2) == (("beta", 1.0), ("zeta", 1.0))
    assert synthetic_vectors.nearest("SOURCE", k=1) == (("beta", 1.0),)
    assert synthetic_vectors.nearest("source", k=1, exclude=("BETA",)) == (("zeta", 1.0),)
    assert synthetic_vectors.nearest("absent") == ()
    assert synthetic_vectors.similarity("source", "absent") is None
    assert synthetic_vectors.nearest("source", k=0) == ()


def test_partitioned_neighbors_match_full_sort_with_ties_and_exclusions():
    import numpy as np

    rng = np.random.default_rng(1207)
    matrix = rng.normal(size=(512, 8)).astype(np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix[1:24] = matrix[0]  # Many equal scores at the top-k boundary.
    words = ["source", *[f"word{i:03d}" for i in range(1, 512)]]
    words[25] = "WORD024"  # Case-folded duplicate must occupy one slot.
    index = VectorIndex(tuple(words), matrix, {"test": "full-sort-equivalence"})

    for query in ("source", "word100"):
        position = index._position(query)
        scores = matrix @ matrix[position]
        for k in (1, 4, 12, 24, 100, 600):
            for excluded_words in ((), tuple(words[1:35]), ("WORD024", "word100")):
                excluded = {word.casefold() for word in excluded_words} | {query.casefold()}
                full_sort = sorted(
                    ((word, max(-1.0, min(1.0, float(scores[i]))))
                     for i, word in enumerate(words) if word.casefold() not in excluded),
                    key=lambda item: (-item[1], item[0].casefold(), item[0]),
                )
                expected = []
                used = set(excluded)
                for word, score in full_sort:
                    if word.casefold() not in used:
                        used.add(word.casefold())
                        expected.append((word, score))
                    if len(expected) == k:
                        break
                assert index.nearest(query, k=k, exclude=excluded_words) == tuple(expected)


def test_plain_glove_format_and_max_words(tmp_path):
    path = tmp_path / "synthetic-glove.txt"
    path.write_text("source 1 0\n\nother 0 3\nlast -1 0\n")
    index = VectorIndex.from_text(path, max_words=2)
    assert index.words == ("source", "other")
    assert index.metadata["source_rows"] == 3
    assert index.metadata["retained_rows"] == 2
    assert index.metadata["format"] == "glove-text"
    assert index.similarity("source", "last") is None
    assert VectorIndex.from_text(path).name != index.name


def test_case_variants_do_not_consume_multiple_generator_slots(tmp_path):
    path = tmp_path / "case-sensitive-vectors.txt"
    path.write_text("source 1 0\nshort 1 0\nSHORT 1 0\nother 0.9 0.1\n")
    index = VectorIndex.from_text(path)
    neighbors = index.nearest("source", k=2)
    assert [word for word, _ in neighbors] == ["SHORT", "other"]
    provider = PilotProvider(
        "b2", vectors=index, phrases={}, max_candidates=4, embedding_slots=2,
    )
    assert [c.replacement for c in provider.candidates("source")] == ["SHORT", "other"]
    assert all(c.provenance.startswith("embedding:") for c in provider.candidates("source"))


@pytest.mark.parametrize("text,match", [
    ("", "no vectors"),
    ("lonely\n", "no coordinates"),
    ("a 1 0\nb 1\n", "dimension"),
    ("a 0 0\n", "Zero vector"),
    ("a nan 1\n", "Nonfinite"),
    ("a inf 1\n", "Nonfinite"),
    ("a 1 broken\n", "coordinate"),
    ("a 1 0\na 0 1\n", "Duplicate"),
    ("2 2\na 1 0\n", "declares"),
    ("1 2\na 1 0\nb 0 1\n", "declares"),
    ("0 2\n", "positive"),
])
def test_malformed_vectors_fail_loudly(tmp_path, text, match):
    path = tmp_path / "malformed.txt"
    path.write_text(text)
    with pytest.raises(ValueError, match=match):
        VectorIndex.from_text(path)


def test_limit_does_not_hide_later_invalid_rows(tmp_path):
    path = tmp_path / "invalid-tail.txt"
    path.write_text("source 1 0\nwrong nan 0\n")
    with pytest.raises(ValueError, match="Nonfinite"):
        VectorIndex.from_text(path, max_words=1)


def test_large_finite_vector_normalizes_without_overflow(tmp_path):
    path = tmp_path / "large-finite.txt"
    path.write_text("source 1e300 1e300\nother 1 1\n")
    assert VectorIndex.from_text(path).similarity("source", "other") == pytest.approx(1)


def test_source_fingerprint_changes_on_contents_not_location(tmp_path):
    path = tmp_path / "source.txt"
    other = tmp_path / "copied.txt"
    path.write_text("source 1 0\nother 0 1\n")
    original = VectorIndex.from_text(path)
    other.write_bytes(path.read_bytes())
    assert original.name == VectorIndex.from_text(other).name
    path.write_text("source 1 0\nother 1 1\n")
    modified = VectorIndex.from_text(path)
    assert original.metadata["sha256"] != modified.metadata["sha256"]
    assert original.name != modified.name


def test_b0_b1_have_identical_full_pool_but_distinct_ranking(fake_wordnet, synthetic_vectors):
    kwargs = dict(vectors=synthetic_vectors, use_wordnet=True, phrases={}, max_candidates=2, embedding_slots=1)
    baseline, ranked = PilotProvider("b0", **kwargs), PilotProvider("b1", **kwargs)
    assert identities(baseline.candidate_pool("source", pos="NOUN")) == identities(ranked.candidate_pool("source", pos="NOUN"))
    assert [c.replacement for c in baseline.candidates("source", pos="NOUN")] == ["alpha", "beta"]
    assert [c.replacement for c in ranked.candidates("source", pos="NOUN")] == ["beta", "zeta"]
    assert ranked.candidate_pool("source")[-1].replacement == "missing"
    assert ranked.candidates("source", pos="VERB") == ()


def test_curated_priority_deduplication_and_fingerprint(fake_wordnet, synthetic_vectors):
    provider = PilotProvider("b1", vectors=synthetic_vectors, use_wordnet=True, phrases={"source": "alpha"})
    pool = provider.candidate_pool("Source", pos="NOUN")
    assert pool[0].replacement == "alpha"
    assert pool[0].provenance == "curated"
    assert len([c for c in pool if c.replacement == "alpha"]) == 1
    assert all(c.source == "Source" for c in pool)
    changed = PilotProvider("b1", vectors=synthetic_vectors, use_wordnet=True, phrases={"source": "beta"})
    assert changed.name != provider.name
    assert changed.metadata["curated"]["sha256"] != provider.metadata["curated"]["sha256"]


def test_b2_fixed_allocation_cannot_crowd_baseline(fake_wordnet, synthetic_vectors):
    provider = PilotProvider("b2", vectors=synthetic_vectors, use_wordnet=True, phrases={}, max_candidates=3, embedding_slots=1)
    result = provider.candidates("source", pos="NOUN")
    assert [c.replacement for c in result] == ["beta", "zeta", "gamma"]
    assert [c.provenance.split(":")[0] for c in result] == ["wordnet", "wordnet", "embedding"]
    assert provider.metadata["baseline_slots"] == 2
    # No WordNet candidates: the generator still receives only its one slot.
    assert len(provider.candidates("source", pos="VERB")) == 1
    assert provider.metadata["unused_slot_policy"] == "no-borrowing"


def test_b2_union_preserves_every_baseline_candidate(fake_wordnet, synthetic_vectors):
    provider = PilotProvider("b2_union", vectors=synthetic_vectors, use_wordnet=True, phrases={}, max_candidates=3, embedding_slots=1)
    pool = provider.candidate_pool("source", pos="NOUN")
    result = provider.candidates("source", pos="NOUN")
    assert result[:len(pool)] == pool
    assert len(result) == len(pool) + 1 > provider.max_candidates
    assert provider.metadata["uncapped_diagnostic"] is True
    assert provider.metadata["union_extra_neighbor_limit"] == 1


def test_provider_memoizes_expensive_search_without_changing_identity(synthetic_vectors, monkeypatch):
    provider = PilotProvider("b2", vectors=synthetic_vectors, phrases={})
    metadata = provider.metadata
    calls = []
    original = synthetic_vectors.nearest

    def counted(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(synthetic_vectors, "nearest", counted)
    first = provider.candidates("source", pos="NOUN")
    assert provider.candidates("source", pos="NOUN") == first
    assert len(calls) == 1
    assert provider.metadata == metadata
    assert len(first) == 4
    # Distinct grammatical queries cannot reuse a different POS pool.
    provider.candidates("source", pos="VERB")
    assert len(calls) == 2


def test_provider_cache_is_bounded():
    provider = PilotProvider(phrases={})
    for i in range(1030):
        provider.candidates(f"absent-{i}")
    assert len(provider._candidate_cache) == len(provider._pool_cache) == 1024
    assert ("absent-0", None) not in provider._candidate_cache


def test_default_rules_and_empty_or_identity_rules():
    provider = PilotProvider()
    assert provider.candidates("in order to")[0].replacement == "to"
    assert provider.max_phrase_words >= 5
    assert provider.candidates("") == ()
    empty = PilotProvider(phrases={"x": "X", "": "a", "b": " "})
    assert empty.metadata["curated"]["rules"] == 0
    assert empty.max_phrase_words == 1


def test_provider_metadata_is_json_and_not_mutable():
    provider = PilotProvider(phrases={"in order to": "to"})
    metadata = provider.metadata
    json.dumps(metadata)
    metadata["curated"]["rules"] = 500
    assert provider.metadata["curated"]["rules"] == 1
    assert provider.metadata["wordnet"]["status"] == "disabled"


@pytest.mark.parametrize("kwargs", [
    {"arm": "wrong"}, {"arm": "b1"}, {"max_candidates": 0},
    {"max_candidates": True}, {"embedding_slots": -1}, {"embedding_slots": 10},
    {"use_wordnet": "yes"}, {"phrases": {"foo": 3}},
    {"phrases": {"FOO": "bar", "foo": "baz"}},
])
def test_invalid_provider_configuration(kwargs):
    with pytest.raises(ValueError):
        PilotProvider(**kwargs)


@pytest.mark.parametrize("embedding_slots", [0, 8])
def test_b2_requires_both_candidate_categories(synthetic_vectors, embedding_slots):
    with pytest.raises(ValueError, match="baseline slot"):
        PilotProvider("b2", vectors=synthetic_vectors, embedding_slots=embedding_slots)


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_retained_word_limit(tmp_path, value):
    with pytest.raises(ValueError, match="max_words"):
        VectorIndex.from_text(tmp_path / "unused", max_words=value)


def test_requested_wordnet_missing_library_is_explicit(monkeypatch):
    monkeypatch.setitem(sys.modules, "nltk", None)
    with pytest.raises(ImportError, match="nltk.downloader wordnet"):
        PilotProvider(use_wordnet=True)


def test_requested_wordnet_missing_corpus_is_explicit(monkeypatch):
    import types

    class MissingCorpus:
        def ensure_loaded(self):
            raise LookupError("synthetic missing local corpus")

    monkeypatch.setitem(sys.modules, "nltk", types.SimpleNamespace(__version__="fake"))
    monkeypatch.setitem(sys.modules, "nltk.corpus", types.SimpleNamespace(wordnet=MissingCorpus()))
    with pytest.raises(RuntimeError, match="local corpus is unavailable"):
        PilotProvider(use_wordnet=True)


def test_wordnet_fingerprint_tracks_contents_independent_of_file_listing_order(monkeypatch):
    import io
    import types

    class LocalCorpus:
        def __init__(self, files):
            self.files = files

        def ensure_loaded(self):
            pass

        def get_version(self):
            return "test-fixture"

        def fileids(self):
            return list(self.files)

        def open(self, file_id):
            return io.StringIO(self.files[file_id])

    def provider_for(files):
        monkeypatch.setitem(sys.modules, "nltk", types.SimpleNamespace(__version__="fake"))
        monkeypatch.setitem(
            sys.modules, "nltk.corpus", types.SimpleNamespace(wordnet=LocalCorpus(files)),
        )
        return PilotProvider(use_wordnet=True, phrases={})

    original = provider_for({"index.noun": "index", "data.noun": "data"})
    reordered = provider_for({"data.noun": "data", "index.noun": "index"})
    modified = provider_for({"index.noun": "index", "data.noun": "changed"})
    assert original.metadata["wordnet"]["status"] == "available"
    assert original.name == reordered.name
    assert original.name != modified.name
    assert original.metadata["wordnet"]["sha256"] != modified.metadata["wordnet"]["sha256"]


def test_vector_missing_numpy_is_actionable(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "numpy", None)
    with pytest.raises(ImportError, match="promptcomp\\[lexical\\]"):
        VectorIndex.from_text(tmp_path / "unused")
