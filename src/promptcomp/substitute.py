"""Stage 4: lossless, dictionary-driven substitution.

Only phrase-shortening (safe, one-way) is implemented here. Entity abbreviation
(bijective, reversible) depends on defined-term extraction from Stage 1/protect
and lands in Plan B (M4-M6).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from importlib.resources import files

import yaml

from .types import Substitution


@dataclass(frozen=True)
class SubstituteResult:
    text: str
    applied: tuple[Substitution, ...]
    version: str


def load_substitutions(path: str | Path | None = None) -> tuple[dict[str, str], str]:
    if path is not None:
        text = Path(path).read_text()
    else:
        text = files("promptcomp").joinpath("data/substitutions.yaml").read_text()
    doc = yaml.safe_load(text)
    return dict(doc.get("phrases", {})), str(doc.get("version", ""))


def substitute(
    text: str,
    phrases: dict[str, str] | None = None,
    *,
    version: str = "",
) -> SubstituteResult:
    if phrases is None:
        phrases, version = load_substitutions()
    if not phrases or not text:
        return SubstituteResult(text=text, applied=(), version=version)

    # Longest keys first so multi-word phrases win over their prefixes.
    keys = sorted(phrases, key=len, reverse=True)
    applied: list[Substitution] = []
    result = text
    for key in keys:
        pattern = re.compile(r"\b" + re.escape(key) + r"\b", re.IGNORECASE)
        if pattern.search(result):
            replacement = phrases[key]
            result, n = pattern.subn(replacement, result)
            if n:
                applied.append(
                    Substitution(original=key, replacement=replacement, reversible=False)
                )
    return SubstituteResult(text=result, applied=tuple(applied), version=version)
