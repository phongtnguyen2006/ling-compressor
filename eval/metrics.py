"""Eval metrics. protected_violation_count reuses the invariant detectors so the
eval's safety number is consistent with the build gate."""
from __future__ import annotations

from collections import Counter

from promptcomp.parse import parse
from promptcomp.protect import (
    extract_negations,
    extract_numbers,
    extract_protected_terms,
    load_protect_list,
)

_PL = load_protect_list()
_NUMERIC_NER_LABELS = {"CARDINAL", "ORDINAL", "QUANTITY", "MONEY", "PERCENT", "DATE", "TIME"}


def _multiset_shortfall(before: Counter, after: Counter) -> int:
    """Count of elements present in `before` but missing/reduced in `after`."""
    missing = before - after
    return sum(missing.values())


def protected_violation_count(original: str, compressed: str) -> int:
    violations = 0
    violations += _multiset_shortfall(
        Counter(extract_numbers(original)), Counter(extract_numbers(compressed))
    )
    violations += _multiset_shortfall(
        Counter(extract_negations(original)), Counter(extract_negations(compressed))
    )
    violations += _multiset_shortfall(
        extract_protected_terms(original, _PL), extract_protected_terms(compressed, _PL)
    )
    # numeric-label named entities (independent oracle)
    for ent in parse(original).ents:
        if ent.label_ in _NUMERIC_NER_LABELS and ent.text.strip() not in compressed:
            violations += 1
    return violations


def parse_failure_rate(text: str) -> float:
    """Fraction of sentences with no ROOT (a parse failure). 0.0 if no sentences."""
    doc = parse(text)
    sents = [s for s in doc.sents if s.text.strip()]
    if not sents:
        return 0.0
    failures = sum(1 for s in sents if not any(t.dep_ == "ROOT" for t in s))
    return failures / len(sents)


def compression_metrics(original: str, compressed: str, counter) -> dict:
    before = counter.count(original)
    after = counter.count(compressed)
    return {
        "tokens_before": before,
        "tokens_after": after,
        "compression_ratio": (after / before) if before else 1.0,
        "token_counter": counter.name,
    }
