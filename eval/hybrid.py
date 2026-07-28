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

import time

from promptcomp import compress as ours_compress
from promptcomp.parse import parse
from promptcomp.protect import (
    extract_negations,
    extract_numbers,
    extract_protected_terms,
    load_protect_list,
)
from promptcomp.segment import segment
from promptcomp.types import BlockKind

from .baselines import Compressed, LLMLingua2Compressor
from .metrics import _NUMERIC_NER_LABELS

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
