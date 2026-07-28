"""Deterministic, CPU-only information-cost scorer.

Lower score => cheaper to delete (less information lost) => deleted first.
Combines four factors, multiplicatively:

  * a per-deprel prior (adjuncts cheap, adjectives/appositives dearer) — a
    hand-tuned analog of Filippova & Strube's syntactic label weight P(l|h);
  * content-word density of the span;
  * corpus rarity of the span's content words (a tf-idf-style lexical-importance
    weight, cf. Filippova & Strube 2008), via `frequency.get_rarity`;
  * position within the block — earlier spans are dearer to delete (a
    deterministic reflection of PartPrompt's leaf-ward first-position boost).

Rarity and position replace the old average-content-word-length rarity proxy,
bringing the implementation in line with COMPRESSOR_SPEC.md §3
("TF-IDF/IDF-weighted content-word density, position, deprel prior").
"""
from __future__ import annotations

import hashlib

from ..types import Candidate
from .frequency import TABLE_VERSION, get_rarity, load_freq_table

# Per-deprel prior: higher => more likely to carry meaning => keep (dearer).
_DEPREL_PRIOR = {
    "discourse": 0.2,
    "advmod": 0.3,
    "prep": 0.4,
    "npadvmod": 0.4,
    "advcl": 0.5,
    "parataxis": 0.6,
    "appos": 0.6,
    "acl": 0.7,
    "relcl": 0.7,
    "nmod": 0.8,
    "amod": 1.0,
}

# Non-prior multiplicative factors are bounded to keep the deprel prior the
# dominant ordering signal: rarity_factor in [1, 1+RARITY_WEIGHT],
# position_factor in [1, 1+POS_BETA].
#
# CONDITIONAL GUARDRAIL — holds only for spans of EQUAL content density:
#     (1 + RARITY_WEIGHT) * (1 + POS_BETA) < prior_amod / prior_advmod
#     (1 + 1.0) * (1 + 0.5) = 3.0  <  1.0 / 0.3 = 3.33   ✓
# i.e. rarity and position together can never reorder an amod below an advmod
# that has the same density. Preserve this inequality when tuning.
#
# The UNCONDITIONAL claim is FALSE and must not be assumed: density_factor
# (= 0.5 + density) has its own 3x swing over [0.5, 1.5] and does NOT cancel
# between two different spans. Worst case is
#     (1.5/0.5) * 2.0 * 1.5 = 9.0  >  3.33,
# so an amod CAN be cheaper than an advmod. Real counterexample:
#     "Idiosyncratically, the auditor reconciled the ledger with other data."
#     amod 'other' = 0.5368  <  advmod 'Idiosyncratically' = 1.3500
# because 'other' is a stopword (density 0). That is acceptable — deleting a
# contentless modifier before a rare sentence-initial adverb is defensible —
# but it is not a guarantee. (Note the previous avg_len term was unbounded, so
# it violated even the conditional form; bounding these factors is what buys
# the conditional guarantee.) See tests/test_scoring.py.
RARITY_WEIGHT = 1.0
POS_BETA = 0.5

# Rarity assigned to a span with no content tokens. Deliberately mid-range, not
# 0.0: density_factor already bottoms out at 0.5 for such spans, and letting
# rarity bottom out too would penalise "has no content words" twice through two
# nominally independent terms, making contentless spans the cheapest thing in
# the document.
NO_CONTENT_RARITY = 0.5


def _prior_digest() -> str:
    """Stable short digest of the deprel prior table, for the version string."""
    payload = repr(sorted(_DEPREL_PRIOR.items())).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:8]


def _token_rarity(tok) -> float:
    """Rarity of a token, preferring its lemma but falling back to the surface
    form when the lemma is out-of-vocabulary.

    spaCy's rule lemmatizer sometimes emits non-words ("tiered" -> "tiere",
    "routed" -> "rout"). Those fall OOV and would score maximally rare, which is
    artificially protective. When that happens the surface form is the better
    estimate. Both lookups are O(1) dict hits.
    """
    table = load_freq_table()
    lemma = (tok.lemma_ or tok.text).lower()
    if lemma in table:
        return table[lemma]
    surface = tok.text.lower()
    if surface in table:
        return table[surface]
    return get_rarity(lemma)  # genuinely OOV -> default (max rarity)


class HeuristicScorer:
    @property
    def name(self) -> str:
        return "heuristic"

    @property
    def version(self) -> str:
        # Encodes every input to the score: the tunable weights, the deprel
        # prior table (hashed — it is the most influential weight set, and
        # editing one entry must not leave config_hash unchanged), and the
        # rarity-table version. Lets weights change without renaming the
        # backend, while keeping runs auditable.
        return (
            f"heuristic-2026.07;rw={RARITY_WEIGHT};pb={POS_BETA}"
            f";nc={NO_CONTENT_RARITY};pri={_prior_digest()};tbl={TABLE_VERSION}"
        )

    def score(self, doc, candidates: list[Candidate]) -> list[float]:
        doc_span = max(1, len(doc.text) - 1)
        scores: list[float] = []
        for cand in candidates:
            toks = [
                t
                for t in doc
                if t.idx >= cand.char_start and (t.idx + len(t.text)) <= cand.char_end
            ]
            content = [t for t in toks if not (t.is_stop or t.is_punct or t.is_space)]
            n = max(1, len(toks))
            density = len(content) / n
            density_factor = 0.5 + density

            if content:
                mean_rarity = sum(_token_rarity(t) for t in content) / len(content)
            else:
                mean_rarity = NO_CONTENT_RARITY
            rarity_factor = 1.0 + RARITY_WEIGHT * mean_rarity

            rel_pos = min(1.0, max(0.0, cand.char_start / doc_span))
            position_factor = 1.0 + POS_BETA * (1.0 - rel_pos)

            prior = _DEPREL_PRIOR.get(cand.deprel, 0.5)
            scores.append(
                round(prior * density_factor * rarity_factor * position_factor, 6)
            )
        return scores
