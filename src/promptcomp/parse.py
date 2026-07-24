"""Stage 1: dependency parsing and deletable-subtree enumeration.

en_core_web_sm emits ClearNLP/OntoNotes dependency labels via token.dep_
(e.g. `prep`, `relcl`, `dobj`, `ROOT`) — NOT Universal Dependencies.
"""
from __future__ import annotations

from .types import Candidate

_NLP = None


def load_nlp():
    """Load and cache the spaCy English pipeline (singleton)."""
    global _NLP
    if _NLP is None:
        try:
            import spacy
        except ImportError as e:
            raise RuntimeError(
                "spaCy is required. Install with: pip install -e '.[dev]' "
                "(spacy is a core dependency)."
            ) from e
        try:
            _NLP = spacy.load("en_core_web_sm")
        except OSError as e:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' is not installed. "
                "Install it with: python -m spacy download en_core_web_sm"
            ) from e
    return _NLP


def parse(text: str):
    """Parse text into a spaCy Doc."""
    return load_nlp()(text)


# spaCy (ClearNLP) adjunct/modifier labels — deletable subtree roots.
SAFE_DEPRELS = frozenset({
    "advmod",    # adverbial modifier
    "advcl",     # adverbial clause
    "acl",       # clausal modifier of noun
    "relcl",     # relative clause modifier
    "appos",     # appositional modifier
    "prep",      # prepositional phrase (UD obl/nmod equivalent)
    "npadvmod",  # noun phrase as adverbial modifier
    "amod",      # adjectival modifier (low priority, but eligible)
    "nmod",      # nominal modifier
    "parataxis",
    "discourse",
})

# spaCy (ClearNLP) core-argument / structural labels — never candidates.
CORE_DEPRELS = frozenset({
    "nsubj", "nsubjpass", "csubj", "csubjpass",
    "dobj", "dative", "iobj",
    "ccomp", "xcomp",
    "ROOT", "aux", "auxpass", "cop",
    "pobj",  # object of a preposition — belongs to its prep's subtree
})


def enumerate_candidates(doc) -> list[Candidate]:
    """Enumerate deletable subtrees (whole subtrees only)."""
    candidates: list[Candidate] = []
    for token in doc:
        if token.dep_ not in SAFE_DEPRELS:
            continue
        subtree = list(token.subtree)
        char_start = min(t.idx for t in subtree)
        char_end = max(t.idx + len(t.text) for t in subtree)
        candidates.append(
            Candidate(
                deprel=token.dep_,
                char_start=char_start,
                char_end=char_end,
                head_text=token.text,
                text=doc.text[char_start:char_end],
                n_tokens=len(subtree),
            )
        )
    candidates.sort(key=lambda c: (c.char_start, c.char_end))
    return candidates
