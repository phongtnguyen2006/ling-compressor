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


def test_scorer_version_reaches_config_hash():
    # Pins the pipeline.py wiring, not just the string: two scorers differing
    # ONLY in version must produce different audit hashes.
    from promptcomp.pipeline import compress

    class _Stub(HeuristicScorer):
        def __init__(self, v):
            self._v = v

        @property
        def version(self):
            return self._v

    text = "The committee approved the annual budget in the morning. " * 3
    h1 = compress(text, min_tokens=0, scorer=_Stub("a")).config_hash
    h2 = compress(text, min_tokens=0, scorer=_Stub("b")).config_hash
    assert h1 != h2


def test_deprel_prior_change_changes_version():
    # The prior table is the most influential weight set; editing it must not
    # leave config_hash unchanged.
    import promptcomp.scoring.heuristic as H

    before = HeuristicScorer().version
    H._DEPREL_PRIOR["amod"] = 0.9
    try:
        assert HeuristicScorer().version != before
    finally:
        H._DEPREL_PRIOR["amod"] = 1.0
    assert HeuristicScorer().version == before


def test_scorer_has_version_encoding_table():
    # version must be non-empty and encode the rarity-table version so a table
    # or weight change flows into the audit config_hash (pipeline reads it via
    # getattr(scorer, "version", "")).
    from promptcomp.scoring.frequency import TABLE_VERSION

    v = HeuristicScorer().version
    assert isinstance(v, str) and v
    assert TABLE_VERSION in v


def _prep_cost(word):
    doc = parse(f"The report was filed in the {word}.")
    cands = [c for c in enumerate_candidates(doc) if c.deprel == "prep"]
    assert cands, "expected a prepositional adjunct candidate"
    return max(HeuristicScorer().score(doc, cands))


def test_rarer_content_is_dearer_to_delete():
    # "system" and "cartel" are both 6 characters, so the two documents have
    # identical length and the candidate sits at an identical char offset. That
    # holds density AND position fixed, leaving corpus rarity as the only
    # difference -- without equal lengths the position term confounds this and
    # the test passes even with the rarity term disabled.
    from promptcomp.scoring.frequency import get_rarity

    assert len("system") == len("cartel")
    assert get_rarity("system") < get_rarity("cartel")
    assert _prep_cost("system") < _prep_cost("cartel")


def test_rarity_term_is_load_bearing():
    # Mutation guard: zeroing RARITY_WEIGHT must collapse the gap the previous
    # test relies on. Without this, deleting the rarity feature entirely would
    # leave the suite green.
    import promptcomp.scoring.heuristic as H

    original = H.RARITY_WEIGHT
    try:
        H.RARITY_WEIGHT = 0.0
        assert _prep_cost("system") == _prep_cost("cartel")
    finally:
        H.RARITY_WEIGHT = original
    assert _prep_cost("system") < _prep_cost("cartel")


def test_contentless_span_is_not_doubly_penalised():
    # A span whose only token is a stopword has density 0. Its rarity must fall
    # back to the neutral midpoint rather than the minimum, otherwise "has no
    # content words" is punished twice and contentless modifiers become the
    # cheapest deletions in the document.
    import promptcomp.scoring.heuristic as H

    doc = parse("Idiosyncratically, the auditor reconciled the ledger with other data.")
    cands = enumerate_candidates(doc)
    scores = dict(zip((c.text for c in cands), HeuristicScorer().score(doc, cands)))
    other = scores.get("other")
    assert other is not None, "expected the stopword amod 'other' as a candidate"
    # 1.0 (density .5 * rarity 1.0 factor) is what the old 0.0 fallback produced.
    floor = 1.0 * (0.5 + 0.0) * (1.0 + H.RARITY_WEIGHT * 0.0)
    assert other > floor


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
