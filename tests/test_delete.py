from promptcomp.parse import parse, enumerate_candidates
from promptcomp.delete import is_contiguous_subtree, apply_deletions, select_deletions
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


def _c(start, end, n):
    return Candidate(deprel="advmod", char_start=start, char_end=end,
                     head_text="x", text="x" * (end - start), n_tokens=n)


def test_select_picks_cheapest_first_until_budget():
    a = _c(0, 5, 1)    # cheap
    b = _c(10, 20, 3)  # dearer position, more tokens
    scored = [(a, 0.1), (b, 0.9)]
    sel = select_deletions(scored, budget_tokens=1, max_fraction=1.0, total_tokens=10)
    assert [c.char_start for c, _ in sel] == [0]  # only the cheapest needed to meet budget


def test_select_respects_max_fraction_cap():
    a = _c(0, 5, 4)
    b = _c(10, 20, 4)
    scored = [(a, 0.1), (b, 0.2)]
    # cap = floor(0.5 * 10) = 5 tokens; a=4 fits, adding b (4 more -> 8) exceeds cap
    sel = select_deletions(scored, budget_tokens=99, max_fraction=0.5, total_tokens=10)
    assert sum(c.n_tokens for c, _ in sel) <= 5
    assert [c.char_start for c, _ in sel] == [0]


def test_select_skips_overlapping_candidates():
    a = _c(0, 20, 3)
    b = _c(5, 10, 1)   # nested inside a
    scored = [(b, 0.1), (a, 0.2)]  # b cheaper, picked first; a overlaps -> skipped
    sel = select_deletions(scored, budget_tokens=99, max_fraction=1.0, total_tokens=100)
    starts = sorted(c.char_start for c, _ in sel)
    assert starts == [5]


def test_select_is_deterministic():
    a = _c(0, 5, 1)
    b = _c(10, 15, 1)
    scored = [(a, 0.5), (b, 0.5)]  # tie -> break by char_start
    sel = select_deletions(scored, budget_tokens=1, max_fraction=1.0, total_tokens=10)
    assert [c.char_start for c, _ in sel] == [0]
