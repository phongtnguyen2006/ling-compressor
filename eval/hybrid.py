"""Prototype: protect-list routing + LLMLingua-2 compression.

Sentence-level segment-and-route (architecture C from the design note):

    sentence contains a protected span?
      YES -> ours   (conservative subtree deletion; protected spans vetoed)
      NO  -> llmlingua2 (compress hard -- there is nothing there to lose)

Safety argument: a sentence containing zero protected tokens *by the same
detectors `protected_violation_count` uses* cannot yield a violation whatever
LLMLingua does to it. So the zero-violation guarantee holds by construction,
not by post-hoc verification.

PASSTHROUGH blocks are copied byte-identical, as everywhere else.
This is a separate named compressor -- deliberately NOT a change to `ours`,
so spec 11.1's "if ours out-compresses llmlingua2, a guardrail leaked" alarm
stays meaningful.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

from promptcomp import compress as ours_compress
from promptcomp.parse import parse
from promptcomp.protect import (
    extract_defined_terms,
    extract_negations,
    extract_numbers,
    extract_protected_terms,
    load_defined_terms,
    load_protect_list,
)
from promptcomp.segment import segment
from promptcomp.types import BlockKind

from .baselines import Compressed, LLMLingua2Compressor
from .metrics import _NUMERIC_NER_LABELS, protected_violation_count

_PL = load_protect_list()
_PROTECTED_NER = _NUMERIC_NER_LABELS | set(_PL.ner_entity)


def sentence_is_protected(sent) -> bool:
    """True if the sentence carries anything the protect-list guards."""
    t = sent.text
    if extract_numbers(t):
        return True
    if extract_negations(t):
        return True
    if extract_protected_terms(t, _PL):
        return True
    return any(e.label_ in _PROTECTED_NER for e in sent.ents)


class HybridCompressor:
    name = "hybrid"

    def __init__(self, device_map: str | None = None) -> None:
        self._ll = LLMLingua2Compressor(device_map=device_map)
        self.stats = {"protected_sents": 0, "clean_sents": 0}

    def _route_block(self, block_text: str, ratio: float) -> str:
        doc = parse(block_text)
        parts: list[str] = []
        pos = 0
        for sent in doc.sents:
            if not sent.text.strip():
                continue
            parts.append(block_text[pos : sent.start_char])  # keep inter-sentence whitespace
            if sentence_is_protected(sent):
                self.stats["protected_sents"] += 1
                parts.append(ours_compress(sent.text, target_ratio=ratio, min_tokens=0).compressed)
            else:
                self.stats["clean_sents"] += 1
                parts.append(self._ll.compress(sent.text, ratio).text)
            pos = sent.end_char
        parts.append(block_text[pos:])
        return "".join(parts)

    def compress(self, text: str, ratio: float) -> Compressed:
        t0 = time.perf_counter()
        out: list[str] = []
        for block in segment(text):
            if block.kind is not BlockKind.PROSE:
                out.append(block.text)  # byte-identical passthrough
            else:
                out.append(self._route_block(block.text, ratio))
        return Compressed(text="".join(out), latency_ms=(time.perf_counter() - t0) * 1000)


# Only independently attached clauses are safe to replace with sentinels. If a
# protected token belongs to the main clause, the whole sentence falls back to
# the conservative compressor.
_ISOLATABLE_CLAUSE_DEPS = frozenset({"advcl", "relcl", "acl", "parataxis"})
_CORE_ANCHOR_DEPS = frozenset({"nsubj", "nsubjpass", "csubj", "csubjpass", "dobj", "iobj", "dative", "attr"})
_NEGATION_FORMS = frozenset({"not", "no", "never", "without", "nor", "neither", "none", "cannot"})
_MODAL_FORMS = frozenset({"shall", "must", "may", "will", "would", "should", "can", "could", "might"})


@dataclass(frozen=True)
class MaskedSentence:
    text: str
    sentinels: tuple[str, ...]
    clauses: tuple[str, ...]
    anchors: tuple[str, ...]
    marker_order: tuple[str, ...]


def _protected_seed_spans(doc, defined_terms=frozenset()) -> list[tuple[int, int]]:
    """Locate exact protected content before expanding it to clause boundaries."""
    spans: set[tuple[int, int]] = set()

    for ent in doc.ents:
        if ent.label_ in _PROTECTED_NER:
            spans.add((ent.start_char, ent.end_char))

    single_terms = set().union(*_PL.single_terms.values())
    for tok in doc:
        forms = {tok.text.lower(), tok.lemma_.lower()}
        is_numberish = tok.like_num or tok.is_currency or tok.text in {"$", "€", "£", "¥", "%"}
        is_negation = tok.dep_ == "neg" or bool(forms & _NEGATION_FORMS) or tok.text.lower().endswith("n't")
        is_modal = tok.dep_ in {"aux", "auxpass"} and bool(forms & _MODAL_FORMS)
        if is_numberish or is_negation or is_modal or forms & single_terms:
            spans.add((tok.idx, tok.idx + len(tok.text)))

    low = doc.text.lower()
    phrases = set().union(*_PL.phrase_terms.values()) | {term.lower() for term in defined_terms}
    for phrase in phrases:
        if not phrase:
            continue
        start = 0
        while True:
            index = low.find(phrase, start)
            if index < 0:
                break
            spans.add((index, index + len(phrase)))
            start = index + len(phrase)

    return sorted(spans)


def _isolating_clause(doc, seed: tuple[int, int]) -> tuple[int, int] | None:
    """Return the smallest contiguous subordinate clause containing a seed."""
    start, end = seed
    seed_tokens = [tok for tok in doc if tok.idx < end and tok.idx + len(tok.text) > start]
    candidates: set[tuple[int, int]] = set()
    for token in seed_tokens:
        for node in (token, *token.ancestors):
            if node.dep_ not in _ISOLATABLE_CLAUSE_DEPS:
                continue
            subtree = list(node.subtree)
            clause_start = min(tok.idx for tok in subtree)
            clause_end = max(tok.idx + len(tok.text) for tok in subtree)
            if clause_start > start or clause_end < end:
                continue
            span = doc.char_span(clause_start, clause_end)
            if span is not None and len(span) == len(subtree):
                candidates.add((clause_start, clause_end))
    return min(candidates, key=lambda item: item[1] - item[0]) if candidates else None


def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def mask_protected_clauses(doc, sent, defined_terms=frozenset()) -> MaskedSentence | None:
    """Mask protected subordinate clauses, or return None when isolation is unsafe."""
    seeds = [
        seed for seed in _protected_seed_spans(doc, defined_terms)
        if seed[0] < sent.end_char and seed[1] > sent.start_char
    ]
    if not seeds:
        return MaskedSentence(sent.text, (), (), (), ())

    clauses: list[tuple[int, int]] = []
    for seed in seeds:
        clause = _isolating_clause(doc, seed)
        if clause is None or clause[0] < sent.start_char or clause[1] > sent.end_char:
            return None
        clauses.append(clause)
    clauses = _merge_spans(clauses)

    parts: list[str] = []
    sentinels: list[str] = []
    originals: list[str] = []
    cursor = sent.start_char
    for index, (start, end) in enumerate(clauses):
        sentinel = f"KEEPCLAUSE{chr(65 + index)}"
        parts.append(doc.text[cursor:start])
        parts.append(sentinel)
        sentinels.append(sentinel)
        originals.append(doc.text[start:end])
        cursor = end
    parts.append(doc.text[cursor:sent.end_char])
    anchor_tokens = tuple(
        token
        for token in sent
        if (token.dep_ == "ROOT" or token.dep_ in _CORE_ANCHOR_DEPS)
        and not any(start <= token.idx < end for start, end in clauses)
    )
    anchors = tuple(token.text for token in anchor_tokens)
    marker_order = tuple(
        marker
        for _, marker in sorted(
            [(token.idx, token.text) for token in anchor_tokens]
            + [(start, sentinel) for (start, _), sentinel in zip(clauses, sentinels)]
        )
    )
    return MaskedSentence(
        "".join(parts), tuple(sentinels), tuple(originals), anchors, marker_order
    )


def _restore_and_validate(original: str, compressed: str, masked: MaskedSentence) -> str | None:
    positions = [compressed.find(sentinel) for sentinel in masked.sentinels]
    if any(position < 0 for position in positions) or positions != sorted(positions):
        return None
    if any(compressed.count(sentinel) != 1 for sentinel in masked.sentinels):
        return None
    lowered = compressed.lower()
    if any(anchor.lower() not in lowered for anchor in masked.anchors):
        return None
    cursor = 0
    for marker in masked.marker_order:
        position = lowered.find(marker.lower(), cursor)
        if position < 0:
            return None
        cursor = position + len(marker)

    restored = compressed
    for sentinel, clause in zip(masked.sentinels, masked.clauses):
        restored = restored.replace(sentinel, clause)
    if protected_violation_count(original, restored) != 0:
        return None
    return restored


class SentinelHybridCompressor:
    """Clause-level hybrid with forced sentinels and fail-closed validation.

    Protected main clauses fall back to `ours`; only independently attached
    protected clauses are masked and routed through LLMLingua-2.
    """

    name = "hybrid_sentinel"

    def __init__(self, device_map: str | None = None, ll_compressor=None) -> None:
        self._ll = ll_compressor or LLMLingua2Compressor(device_map=device_map)
        self.stats = {
            "clean_sents": 0,
            "masked_sents": 0,
            "fallback_sents": 0,
            "validation_fallbacks": 0,
            "protected_clauses": 0,
        }

    def _route_block(self, block_text: str, ratio: float, defined_terms) -> str:
        doc = parse(block_text)
        parts: list[str] = []
        cursor = 0
        for sent in doc.sents:
            if not sent.text.strip():
                continue
            parts.append(block_text[cursor:sent.start_char])
            masked = mask_protected_clauses(doc, sent, defined_terms)
            if masked is None:
                self.stats["fallback_sents"] += 1
                parts.append(ours_compress(sent.text, target_ratio=ratio, min_tokens=0).compressed)
            elif not masked.sentinels:
                self.stats["clean_sents"] += 1
                parts.append(self._ll.compress(sent.text, ratio).text)
            else:
                self.stats["masked_sents"] += 1
                self.stats["protected_clauses"] += len(masked.sentinels)
                forced = (*masked.sentinels, *masked.anchors)
                out = self._ll.compress_forced(masked.text, ratio, forced)
                restored = _restore_and_validate(sent.text, out.text, masked)
                if restored is None:
                    self.stats["validation_fallbacks"] += 1
                    restored = ours_compress(sent.text, target_ratio=ratio, min_tokens=0).compressed
                parts.append(restored)
            cursor = sent.end_char
        parts.append(block_text[cursor:])
        return "".join(parts)

    def compress(self, text: str, ratio: float) -> Compressed:
        t0 = time.perf_counter()
        configured_terms, _ = load_defined_terms()
        defined_terms = frozenset(configured_terms) | frozenset(extract_defined_terms(text))
        output: list[str] = []
        for block in segment(text):
            if block.kind is not BlockKind.PROSE:
                output.append(block.text)
            else:
                output.append(self._route_block(block.text, ratio, defined_terms))
        return Compressed(
            text="".join(output),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )
