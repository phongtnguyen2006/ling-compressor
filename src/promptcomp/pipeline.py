from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from .delete import apply_deletions, is_contiguous_subtree, select_deletions
from .parse import enumerate_candidates, parse
from .protect import (
    extract_defined_terms,
    load_defined_terms,
    load_protect_list,
    veto,
)
from .scoring.heuristic import HeuristicScorer
from .segment import segment
from .substitute import load_substitutions
from .substitute import substitute as apply_substitution
from .tokens import CachedCounter, TiktokenCounter, TokenCounter
from .types import BlockKind, CompressionResult, Span, Veto


def _config_hash(**kwargs) -> str:
    payload = json.dumps(kwargs, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def compress(
    text: str,
    *,
    target_ratio: float = 0.6,
    scorer=None,
    substitute: bool = True,
    min_tokens: int = 500,
    max_deletion_fraction: float = 0.5,
    counter: TokenCounter | None = None,
    cache_path: str | Path | None = None,
) -> CompressionResult:
    t0 = time.perf_counter()
    timings: dict[str, float] = {}

    if counter is None:
        counter = CachedCounter(TiktokenCounter(), cache_path or ".token_cache.sqlite")
    if scorer is None:
        scorer = HeuristicScorer()

    protect_list = load_protect_list()
    phrases, sub_version = load_substitutions() if substitute else ({}, "")

    cfg = dict(
        target_ratio=target_ratio,
        substitute=substitute,
        min_tokens=min_tokens,
        max_deletion_fraction=max_deletion_fraction,
        scorer=scorer.name,
        protect_list_version=protect_list.version,
        sub_version=sub_version,
        counter=counter.name,
    )
    config_hash = _config_hash(**cfg)

    tokens_before = counter.count(text)

    if tokens_before < min_tokens:
        timings["total"] = (time.perf_counter() - t0) * 1000
        return CompressionResult(
            original=text, compressed=text, tokens_before=tokens_before,
            tokens_after=tokens_before, token_counter=counter.name, ratio=1.0,
            deleted=(), vetoed=(), substitutions={},
            protect_list_version=protect_list.version, config_hash=config_hash,
            timings_ms=timings, warnings=(),
        )

    yaml_terms, _ = load_defined_terms()
    defined = frozenset(yaml_terms) | frozenset(extract_defined_terms(text))

    out_parts: list[str] = []
    deleted: list = []
    vetoed: list = []
    applied_map: dict[str, str] = {}
    warnings: list[str] = []

    for block in segment(text):
        if block.kind is not BlockKind.PROSE:
            out_parts.append(block.text)
            continue

        doc = parse(block.text)
        cands = enumerate_candidates(doc)
        survivors, block_vetoes = veto(doc, cands, protect_list, defined)
        for v in block_vetoes:
            vetoed.append(
                Veto(
                    span=Span(v.span.start + block.start, v.span.end + block.start),
                    text=v.text, protect_class=v.protect_class,
                )
            )

        deletable = [c for c in survivors if is_contiguous_subtree(doc, c)]
        scores = scorer.score(doc, deletable)
        scored = list(zip(deletable, scores))
        total = len(doc)
        budget = int((1 - target_ratio) * total)
        selected = select_deletions(scored, budget, max_deletion_fraction, total)

        new_text, dels = apply_deletions(block.text, selected, base_offset=block.start)
        deleted.extend(dels)

        if substitute:
            res = apply_substitution(new_text, phrases, version=sub_version)
            new_text = res.text
            for s in res.applied:
                applied_map[s.original] = s.replacement

        out_parts.append(new_text)

    compressed = "".join(out_parts)
    tokens_after = counter.count(compressed)
    timings["total"] = (time.perf_counter() - t0) * 1000

    return CompressionResult(
        original=text, compressed=compressed, tokens_before=tokens_before,
        tokens_after=tokens_after, token_counter=counter.name,
        ratio=(tokens_after / tokens_before) if tokens_before else 1.0,
        deleted=tuple(deleted), vetoed=tuple(vetoed), substitutions=applied_map,
        protect_list_version=protect_list.version, config_hash=config_hash,
        timings_ms=timings, warnings=tuple(warnings),
    )
