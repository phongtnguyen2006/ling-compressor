"""Stage 1: dependency parsing and deletable-subtree enumeration.

en_core_web_sm emits ClearNLP/OntoNotes dependency labels via token.dep_
(e.g. `prep`, `relcl`, `dobj`, `ROOT`) — NOT Universal Dependencies.
"""
from __future__ import annotations

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
