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


def test_scorer_has_version_encoding_table():
    # version must be non-empty and encode the rarity-table version so a table
    # or weight change flows into the audit config_hash (pipeline reads it via
    # getattr(scorer, "version", "")).
    from promptcomp.scoring.frequency import TABLE_VERSION

    v = HeuristicScorer().version
    assert isinstance(v, str) and v
    assert TABLE_VERSION in v


def test_rarer_content_is_dearer_to_delete():
    # Same deprel, structure, and position; only the object noun's corpus rarity
    # differs. The rarer word must make its span dearer (higher cost) to delete.
    s = HeuristicScorer()

    def prep_cost(word):
        doc = parse(f"The report was filed in the {word}.")
        cands = [c for c in enumerate_candidates(doc) if c.deprel == "prep"]
        assert cands, "expected a prepositional adjunct candidate"
        return max(s.score(doc, cands))

    assert prep_cost("area") < prep_cost("reticulum")


def test_earlier_position_is_dearer_to_delete():
    # The same adverbial adjunct is dearer to delete when it appears earlier in
    # the block (PartPrompt-style first-position boost).
    s = HeuristicScorer()

    def quietly_cost(text):
        doc = parse(text)
        cands = [c for c in enumerate_candidates(doc) if c.text.strip(", ").lower() == "quietly"]
        assert cands, "expected a 'quietly' adjunct candidate"
        return max(s.score(doc, cands))

    early = quietly_cost("Quietly, the committee approved the annual budget after review.")
    late = quietly_cost("The committee approved the annual budget after review, quietly.")
    assert early > late
