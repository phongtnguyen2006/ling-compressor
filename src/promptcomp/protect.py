"""Stage 2: protect-list veto.

Strikes any candidate subtree that contains — or is syntactically scoped by —
a protected token. This is a hard rule: over-protecting is safe, under-protecting
is a build failure.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
import re

import yaml

from .types import Candidate, Span, Veto

_MODAL_AUX_WORDS = frozenset({"shall", "must", "may", "will", "would", "should", "can", "could", "might"})

# Currency/optional-thousands/optional-decimal/optional-percent numeric literal.
# No trailing-comma absorption; catches a leading sign and leading decimals.
_NUMBER_RE = re.compile(r"[-−]?\$?(?:\d[\d,]*\d|\d)(?:\.\d+)?%?|\.\d+%?")

# Negation words treated as protected. Matched as whole words, case-insensitive,
# plus the contracted negative suffix n't / n’t (isn't, won't, can't, doesn't).
_NEGATION_WORDS = frozenset(
    {"not", "no", "never", "without", "nor", "neither", "none", "cannot"}
)
_NEGATION_RE = re.compile(
    r"\b(?:not|no|never|without|nor|neither|none|cannot)\b|n['’]t",
    re.IGNORECASE,
)

# "X" means ... / "X" means ... (accepts both straight and curly quotes)
_DEF_MEANS_RE = re.compile(r'["“]([A-Z][^"”]{1,60})["”]\s+means\b')
# X ("the Y") / X ("Y") (accepts both straight and curly quotes)
_DEF_PAREN_RE = re.compile(r'\(["“](?:the\s+)?([A-Z][^"”]{1,60})["”]\)')


@dataclass(frozen=True)
class ProtectList:
    version: str
    ner_numeric: frozenset[str]
    ner_entity: frozenset[str]
    single_terms: dict[str, frozenset[str]]
    phrase_terms: dict[str, frozenset[str]]


def _read_resource(name: str, path: str | Path | None) -> dict:
    if path is not None:
        text = Path(path).read_text()
    else:
        text = files("promptcomp").joinpath(f"data/{name}").read_text()
    return yaml.safe_load(text)


def load_protect_list(path: str | Path | None = None) -> ProtectList:
    doc = _read_resource("protect_terms.yaml", path)
    single = {k: frozenset(t.lower() for t in v) for k, v in doc.get("single_terms", {}).items()}
    phrase = {k: frozenset(t.lower() for t in v) for k, v in doc.get("phrase_terms", {}).items()}
    return ProtectList(
        version=str(doc["version"]),
        ner_numeric=frozenset(doc.get("ner_numeric", [])),
        ner_entity=frozenset(doc.get("ner_entity", [])),
        single_terms=single,
        phrase_terms=phrase,
    )


def load_defined_terms(path: str | Path | None = None) -> tuple[frozenset[str], str]:
    doc = _read_resource("defined_terms.yaml", path)
    return frozenset(doc.get("terms", []) or []), str(doc.get("version", ""))


def extract_numbers(text: str) -> list[str]:
    """Every numeric literal in document order (the invariant multiset)."""
    return _NUMBER_RE.findall(text)


def extract_negations(text: str) -> list[str]:
    """Every negation-word occurrence, lowercased, in document order."""
    return [m.group(0).lower() for m in _NEGATION_RE.finditer(text)]


def extract_protected_terms(text: str, protect_list: ProtectList) -> Counter:
    """Multiset of protect-list term occurrences (single words + phrases), lowercased.

    Mirrors the veto's term matching so the invariant suite and the veto agree.
    """
    low = text.lower()
    counts: Counter = Counter()
    # single-word terms: whole-word, case-insensitive
    for words in protect_list.single_terms.values():
        for w in words:
            n = len(re.findall(r"\b" + re.escape(w) + r"\b", low))
            if n:
                counts[w] += n
    # phrase terms: substring occurrences (non-overlapping)
    for phrases in protect_list.phrase_terms.values():
        for p in phrases:
            if not p:
                continue
            start = 0
            c = 0
            while True:
                i = low.find(p, start)
                if i == -1:
                    break
                c += 1
                start = i + len(p)
            if c:
                counts[p] += c
    return counts


def extract_defined_terms(text: str) -> set[str]:
    """Terms introduced by definition patterns; supplements the yaml list."""
    terms: set[str] = set()
    for pat in (_DEF_MEANS_RE, _DEF_PAREN_RE):
        for m in pat.finditer(text):
            terms.add(m.group(1).strip())
    return terms


def _span_tokens(doc, start: int, end: int) -> list:
    return [t for t in doc if t.idx >= start and (t.idx + len(t.text)) <= end]


def _content_class(doc, cand: Candidate, pl: ProtectList):
    text = cand.text

    if extract_numbers(text):
        return "number"

    for ent in doc.ents:
        if ent.label_ in pl.ner_numeric or ent.label_ in pl.ner_entity:
            if not (ent.end_char <= cand.char_start or ent.start_char >= cand.char_end):
                return "number" if ent.label_ in pl.ner_numeric else "entity"

    toks = _span_tokens(doc, cand.char_start, cand.char_end)

    # Currency/percent symbols and number-adjacent fragments belong to their
    # number and must never be severed from it.
    if any(t.is_currency or t.text in {"$", "€", "£", "¥", "%"} for t in toks):
        return "number"
    before = doc.text[cand.char_start - 1] if cand.char_start > 0 else ""
    after = doc.text[cand.char_end] if cand.char_end < len(doc.text) else ""
    if before.isdigit() or after.isdigit():
        return "number"

    if any(t.dep_ == "neg" for t in toks) or extract_negations(text):
        return "negation"

    forms = {t.text.lower() for t in toks} | {t.lemma_.lower() for t in toks}
    for cls, words in pl.single_terms.items():
        if forms & words:
            return cls

    # Modal auxiliaries anywhere in the span (consistent with the scope rule's
    # _MODAL_AUX_WORDS; catches modals nested deeper than a direct child).
    for t in toks:
        if t.dep_ in {"aux", "auxpass"} and t.text.lower() in _MODAL_AUX_WORDS:
            return "modal"

    return None


def _find_phrase_spans(text: str, protect_list: ProtectList) -> list[tuple[int, int, str]]:
    low = text.lower()
    spans: list[tuple[int, int, str]] = []
    for cls, phrases in protect_list.phrase_terms.items():
        for phrase in phrases:
            if not phrase:
                continue
            start = 0
            while True:
                i = low.find(phrase, start)
                if i == -1:
                    break
                spans.append((i, i + len(phrase), cls))
                start = i + 1
    return spans


def _find_defined_spans(text: str, defined_terms: frozenset[str]) -> list[tuple[int, int, str]]:
    low = text.lower()
    spans: list[tuple[int, int, str]] = []
    for term in defined_terms:
        t = term.lower()
        if not t:
            continue
        start = 0
        while True:
            i = low.find(t, start)
            if i == -1:
                break
            spans.append((i, i + len(t), "defined_term"))
            start = i + 1
    return spans


def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return not (a_end <= b_start or b_end <= a_start)


def _in_neg_or_modal_scope(root_token) -> bool:
    chain = [root_token] + list(root_token.ancestors)
    for anc in chain:
        for child in anc.children:
            if child.dep_ == "neg":
                return True
            if child.dep_ in {"aux", "auxpass"} and child.text.lower() in _MODAL_AUX_WORDS:
                return True
    return False


def veto(doc, candidates, protect_list, defined_terms=frozenset()):
    """Partition candidates into (survivors, vetoes). Hard rule; deterministic."""
    phrase_spans = _find_phrase_spans(doc.text, protect_list)
    defined_spans = _find_defined_spans(doc.text, defined_terms)
    survivors: list[Candidate] = []
    vetoes: list[Veto] = []
    for cand in candidates:
        span = doc.char_span(cand.char_start, cand.char_end)
        reason = None
        if span is None:
            reason = "unaligned"
        else:
            reason = _content_class(doc, cand, protect_list)
            if reason is None:
                for ps, pe, cls in phrase_spans:
                    if _overlaps(cand.char_start, cand.char_end, ps, pe):
                        reason = cls
                        break
            if reason is None:
                for ds, de, cls in defined_spans:
                    if _overlaps(cand.char_start, cand.char_end, ds, de):
                        reason = cls
                        break
            if reason is None and _in_neg_or_modal_scope(span.root):
                reason = "scope"
        if reason is None:
            survivors.append(cand)
        else:
            vetoes.append(
                Veto(
                    span=Span(cand.char_start, cand.char_end),
                    text=cand.text,
                    protect_class=reason,
                )
            )
    return survivors, vetoes
