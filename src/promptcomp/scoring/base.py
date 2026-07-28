from __future__ import annotations

from typing import Protocol

from ..types import Candidate


class Scorer(Protocol):
    @property
    def name(self) -> str: ...
    def score(self, doc, candidates: list[Candidate]) -> list[float]: ...


# Optional, by convention: a scorer MAY expose a ``version: str`` property that
# encodes its weights/data-table so changes flow into the audit ``config_hash``.
# The pipeline reads it via ``getattr(scorer, "version", "")`` so scorers that
# omit it still satisfy this Protocol.
