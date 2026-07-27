from promptcomp.parse import parse, enumerate_candidates
from promptcomp.scoring.heuristic import HeuristicScorer


def _scored(text):
    doc = parse(text)
    cands = enumerate_candidates(doc)
    scores = HeuristicScorer().score(doc, cands)
    return cands, scores


def test_scores_align_with_candidates():
    cands, scores = _scored("The committee approved the annual budget in the morning.")
    assert len(scores) == len(cands)
    assert all(isinstance(s, float) for s in scores)


def test_scorer_is_deterministic():
    doc = parse("The committee quickly approved the annual budget in the morning.")
    cands = enumerate_candidates(doc)
    s1 = HeuristicScorer().score(doc, cands)
    s2 = HeuristicScorer().score(doc, cands)
    assert s1 == s2


def test_adverb_cheaper_than_adjective():
    # a bare discourse/adverbial adjunct should be cheaper to delete than an
    # adjectival modifier (which carries content).
    doc = parse("The remarkably thorough committee approved the budget quietly.")
    cands = enumerate_candidates(doc)
    scores = dict(zip((c.text for c in cands), HeuristicScorer().score(doc, cands)))
    # "quietly" (advmod) should score lower (cheaper) than "remarkably thorough" (amod subtree)
    advmods = [v for k, v in scores.items() if k == "quietly"]
    amods = [v for k, v in scores.items() if "thorough" in k]
    assert advmods and amods
    assert min(advmods) < max(amods)


def test_scorer_has_name():
    assert HeuristicScorer().name == "heuristic"
