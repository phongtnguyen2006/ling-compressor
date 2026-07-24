from promptcomp.parse import parse, load_nlp, enumerate_candidates, SAFE_DEPRELS, CORE_DEPRELS
from promptcomp.types import Candidate


def test_parse_splits_sentences_and_has_root():
    doc = parse("The cat sat. The dog ran fast.")
    sents = list(doc.sents)
    assert len(sents) == 2
    assert any(t.dep_ == "ROOT" for t in doc)


def test_load_nlp_is_cached_singleton():
    assert load_nlp() is load_nlp()


def test_parse_preserves_text():
    text = "She quickly finished the difficult task."
    doc = parse(text)
    assert doc.text == text


def test_prep_phrase_is_candidate():
    doc = parse("The committee approved the budget in the morning.")
    cands = enumerate_candidates(doc)
    assert any("in the morning" in c.text for c in cands)


def test_core_arguments_are_never_candidates():
    doc = parse("The committee approved the budget in the morning.")
    cands = enumerate_candidates(doc)
    for c in cands:
        assert c.deprel not in CORE_DEPRELS
    # subject head "committee" and object head "budget" must not head a candidate
    heads = {c.head_text.lower() for c in cands}
    assert "committee" not in heads
    assert "budget" not in heads


def test_relative_clause_is_candidate():
    doc = parse("The report that arrived late was incomplete.")
    cands = enumerate_candidates(doc)
    assert any("that arrived late" in c.text for c in cands)


def test_adverb_is_candidate():
    doc = parse("She quickly finished the task.")
    cands = enumerate_candidates(doc)
    assert any(c.text == "quickly" for c in cands)


def test_candidate_char_ranges_slice_to_text():
    doc = parse("The committee approved the budget in the morning after long debate.")
    cands = enumerate_candidates(doc)
    assert cands  # non-empty
    for c in cands:
        assert doc.text[c.char_start:c.char_end] == c.text
        assert c.n_tokens >= 1


def test_enumeration_is_deterministic_and_sorted():
    doc = parse("The committee approved the budget in the morning after long debate.")
    a = enumerate_candidates(doc)
    b = enumerate_candidates(doc)
    assert a == b
    assert a == sorted(a, key=lambda c: (c.char_start, c.char_end))


def test_safe_and_core_are_disjoint():
    assert SAFE_DEPRELS.isdisjoint(CORE_DEPRELS)
