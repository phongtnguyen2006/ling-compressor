from __future__ import annotations

from typing import Protocol

from ..types import Candidate


class Scorer(Protocol):
    @property
    def name(self) -> str: ...
    def score(self, doc, candidates: list[Candidate]) -> list[float]: ...
