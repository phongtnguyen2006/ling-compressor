"""Stage 3 deletion mechanics: contiguity guard, range removal, seam cleanup."""
from __future__ import annotations

from .types import Candidate, Deletion, Span

# Punctuation that must hug the preceding word (no space before it).
_LEAD_PUNCT = frozenset(".,;:!?)]}%")
# If a dangling comma is immediately followed by one of these, drop the comma.
_COMMA_ABSORB = frozenset(",.;:!?)]}")


def is_contiguous_subtree(doc, cand: Candidate) -> bool:
    """True iff the candidate's char range is exactly its dependency subtree."""
    span = doc.char_span(cand.char_start, cand.char_end)
    return span is not None and len(span) == cand.n_tokens


def _join_seam(left: str, right: str) -> str:
    """Join two kept pieces across a deletion cut, cleaning only the junction.

    Touches only whitespace/punctuation adjacent to the seam; never rewrites
    formatting elsewhere in either piece.
    """
    l = left.rstrip(" \t")
    r = right.lstrip(" \t")
    had_space = (left != l) or (right != r)
    # Orphaned comma left behind by a deleted clause: drop it if the surviving
    # right side begins with terminal/close/other punctuation (incl. another comma).
    if l.endswith(",") and r[:1] in _COMMA_ABSORB:
        l = l[:-1]
    if not r:
        return l
    if r[0] in _LEAD_PUNCT:
        return l + r
    if l[-1:] in "([{":
        return l + r
    if had_space:
        return l + " " + r
    return l + r


def apply_deletions(
    text: str,
    selected: list[tuple[Candidate, float]],
    base_offset: int = 0,
) -> tuple[str, list[Deletion]]:
    if not selected:
        return text, []
    ordered = sorted(selected, key=lambda cs: cs[0].char_start)
    kept: list[str] = []
    deletions: list[Deletion] = []
    pos = 0
    for cand, sc in ordered:
        if cand.char_start < pos:
            raise ValueError("apply_deletions received overlapping candidates")
        kept.append(text[pos : cand.char_start])
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
    kept.append(text[pos:])
    result = kept[0]
    for piece in kept[1:]:
        result = _join_seam(result, piece)
    return result, deletions


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
