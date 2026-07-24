"""Stage 3 deletion mechanics: contiguity guard, range removal, seam cleanup."""
from __future__ import annotations

import re

from .types import Candidate, Deletion, Span

# Whitespace-seam normalizers applied after removing spans.
_MULTISPACE = re.compile(r"[ \t]{2,}")
_SPACE_BEFORE_PUNCT = re.compile(r"[ \t]+([,.;:!?)\]])")
_SPACE_AFTER_OPEN = re.compile(r"([(\[])[ \t]+")
_SPACE_BEFORE_NL = re.compile(r"[ \t]+(\n)")
_SPACE_AFTER_NL = re.compile(r"(\n)[ \t]+")
_DOUBLE_COMMA = re.compile(r",\s*,")
_SPACE_RUN = None  # reserved


def is_contiguous_subtree(doc, cand: Candidate) -> bool:
    """True iff the candidate's char range is exactly its dependency subtree."""
    span = doc.char_span(cand.char_start, cand.char_end)
    return span is not None and len(span) == cand.n_tokens


def _normalize_seams(text: str) -> str:
    text = _DOUBLE_COMMA.sub(",", text)
    text = _MULTISPACE.sub(" ", text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _SPACE_AFTER_OPEN.sub(r"\1", text)
    text = _SPACE_BEFORE_NL.sub(r"\1", text)
    text = _SPACE_AFTER_NL.sub(r"\1", text)
    text = _MULTISPACE.sub(" ", text)
    return text


def apply_deletions(
    text: str,
    selected: list[tuple[Candidate, float]],
    base_offset: int = 0,
) -> tuple[str, list[Deletion]]:
    if not selected:
        return text, []
    ordered = sorted(selected, key=lambda cs: cs[0].char_start)
    out: list[str] = []
    deletions: list[Deletion] = []
    pos = 0
    for cand, sc in ordered:
        out.append(text[pos : cand.char_start])
        deletions.append(
            Deletion(
                span=Span(cand.char_start + base_offset, cand.char_end + base_offset),
                text=text[cand.char_start : cand.char_end],
                deprel=cand.deprel,
                score=sc,
                reason="budget",
            )
        )
        pos = cand.char_end
    out.append(text[pos:])
    return _normalize_seams("".join(out)), deletions


def _overlaps(a: Candidate, b: Candidate) -> bool:
    return not (a.char_end <= b.char_start or b.char_end <= a.char_start)


def select_deletions(
    scored: list[tuple[Candidate, float]],
    budget_tokens: int,
    max_fraction: float,
    total_tokens: int,
) -> list[tuple[Candidate, float]]:
    cap = int(max_fraction * total_tokens)
    order = sorted(scored, key=lambda cs: (cs[1], cs[0].char_start))
    selected: list[tuple[Candidate, float]] = []
    removed = 0
    for cand, sc in order:
        if removed >= budget_tokens:
            break
        if removed + cand.n_tokens > cap:
            continue
        if any(_overlaps(cand, chosen) for chosen, _ in selected):
            continue
        selected.append((cand, sc))
        removed += cand.n_tokens
    return selected
