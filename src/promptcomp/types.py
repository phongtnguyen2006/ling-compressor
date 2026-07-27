from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class BlockKind(str, Enum):
    PROSE = "prose"
    PASSTHROUGH = "passthrough"


@dataclass(frozen=True)
class Span:
    start: int
    end: int

    def text(self, source: str) -> str:
        return source[self.start : self.end]


@dataclass(frozen=True)
class Block:
    kind: BlockKind
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class Deletion:
    span: Span
    text: str
    deprel: str
    score: float
    reason: str


@dataclass(frozen=True)
class Veto:
    span: Span
    text: str
    protect_class: str


@dataclass(frozen=True)
class Substitution:
    original: str
    replacement: str
    reversible: bool


@dataclass(frozen=True)
class Candidate:
    deprel: str
    char_start: int
    char_end: int
    head_text: str
    text: str
    n_tokens: int


@dataclass(frozen=True)
class CompressionResult:
    original: str
    compressed: str
    tokens_before: int
    tokens_after: int
    token_counter: str
    ratio: float
    deleted: tuple[Deletion, ...]
    vetoed: tuple[Veto, ...]
    substitutions: dict[str, str]
    protect_list_version: str
    config_hash: str
    timings_ms: dict[str, float]
    warnings: tuple[str, ...] = field(default_factory=tuple)
