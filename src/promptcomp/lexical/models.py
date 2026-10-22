"""Small dependency-free contracts shared by lexical pilot components."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class LexicalCandidate:
    """A proposed replacement; provenance and similarity are not fidelity claims."""

    source: str
    replacement: str
    provenance: str
    similarity: float | None = None
    pos: str | None = None


class CandidateProvider(Protocol):
    name: str
    max_phrase_words: int

    def candidates(
        self, source: str, *, pos: str | None = None
    ) -> tuple[LexicalCandidate, ...]:
        """Return ordered candidates for one occurrence's source and POS."""
        ...
