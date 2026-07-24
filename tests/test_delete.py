from promptcomp.parse import parse, enumerate_candidates
from promptcomp.delete import is_contiguous_subtree, apply_deletions
from promptcomp.types import Candidate


def test_contiguous_subtree_true_for_projective_pp():
    doc = parse("The committee approved the budget in the morning.")
    cands = enumerate_candidates(doc)
    pp = next(c for c in cands if "in the morning" in c.text)
    assert is_contiguous_subtree(doc, pp) is True


def test_apply_deletions_removes_range_and_normalizes_seam():
    text = "The committee approved the budget in the morning."
    doc = parse(text)
    cands = enumerate_candidates(doc)
    pp = next(c for c in cands if "in the morning" in c.text)
    out, deletions = apply_deletions(text, [(pp, 1.0)])
    assert "in the morning" not in out
    assert "  " not in out                 # no double space at the seam
    assert out == "The committee approved the budget."   # clean, grammatical remnant
    assert len(deletions) == 1
    assert deletions[0].text == "in the morning"
    assert deletions[0].deprel == pp.deprel


def test_apply_deletions_empty_is_identity():
    text = "Nothing to delete here."
    out, deletions = apply_deletions(text, [])
    assert out == text
    assert deletions == []


def test_apply_deletions_base_offset_shifts_spans():
    text = "The committee approved the budget in the morning."
    doc = parse(text)
    cands = enumerate_candidates(doc)
    pp = next(c for c in cands if "in the morning" in c.text)
    _, deletions = apply_deletions(text, [(pp, 1.0)], base_offset=100)
    assert deletions[0].span.start == pp.char_start + 100
    assert deletions[0].span.end == pp.char_end + 100
