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

from ..types import Candidate
from .frequency import TABLE_VERSION, get_rarity

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

# Non-prior multiplicative factors are bounded so the deprel prior always
# dominates ordering. rarity_factor in [1, 1+RARITY_WEIGHT]; position_factor in
# [1, 1+POS_BETA]. GUARDRAIL: to guarantee no amod span (prior 1.0) is ever
# cheaper than any advmod span (prior 0.3), keep the combined non-prior swing
# under the prior ratio:
#     (1 + RARITY_WEIGHT) * (1 + POS_BETA) < 1 / min_cheap_prior
#     (1 + 1.0) * (1 + 0.5) = 3.0  <  1 / 0.3 = 3.33   ✓
# Any change to these constants MUST preserve that inequality
# (see tests/test_scoring.py::test_adverb_cheaper_than_adjective).
RARITY_WEIGHT = 1.0
POS_BETA = 0.5


class HeuristicScorer:
    @property
    def name(self) -> str:
        return "heuristic"

    @property
    def version(self) -> str:
        # Encodes the scoring weights and the rarity-table version so any change
        # flows into the audit config_hash without renaming the backend.
        return f"heuristic-2026.07;rw={RARITY_WEIGHT};pb={POS_BETA};tbl={TABLE_VERSION}"

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
                mean_rarity = sum(get_rarity(t.lemma_ or t.text) for t in content) / len(content)
            else:
                mean_rarity = 0.0
            rarity_factor = 1.0 + RARITY_WEIGHT * mean_rarity

            rel_pos = min(1.0, max(0.0, cand.char_start / doc_span))
            position_factor = 1.0 + POS_BETA * (1.0 - rel_pos)

            prior = _DEPREL_PRIOR.get(cand.deprel, 0.5)
            scores.append(
                round(prior * density_factor * rarity_factor * position_factor, 6)
            )
        return scores
