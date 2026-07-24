"""Deterministic, CPU-only information-cost scorer.

Lower score => cheaper to delete (less information lost) => deleted first.
Combines content-word density, average content-word length (a rarity proxy),
and a per-deprel prior (adjuncts cheap, adjectives/appositives dearer).
"""
from __future__ import annotations

from ..types import Candidate

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


class HeuristicScorer:
    @property
    def name(self) -> str:
        return "heuristic"

    def score(self, doc, candidates: list[Candidate]) -> list[float]:
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
            avg_len = (sum(len(t.text) for t in content) / len(content)) if content else 0.0
            prior = _DEPREL_PRIOR.get(cand.deprel, 0.5)
            scores.append(round(prior * (0.5 + density) * (1.0 + avg_len / 10.0), 6))
        return scores
