from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from .segment import segment
from .substitute import load_substitutions
from .substitute import substitute as apply_substitution
from .tokens import CachedCounter, TiktokenCounter, TokenCounter
from .types import BlockKind, CompressionResult


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
        counter = CachedCounter(
            TiktokenCounter(), cache_path or ".token_cache.sqlite"
        )

    cfg = dict(
        target_ratio=target_ratio,
        substitute=substitute,
        min_tokens=min_tokens,
        max_deletion_fraction=max_deletion_fraction,
    )
    config_hash = _config_hash(**cfg)

    tokens_before = counter.count(text)

    # Size gate: below the floor, return unchanged.
    if tokens_before < min_tokens:
        timings["total"] = (time.perf_counter() - t0) * 1000
        return CompressionResult(
            original=text,
            compressed=text,
            tokens_before=tokens_before,
            tokens_after=tokens_before,
            token_counter=counter.name,
            ratio=1.0,
            deleted=(),
            vetoed=(),
            substitutions={},
            protect_list_version="none",
            config_hash=config_hash,
            timings_ms=timings,
            warnings=(),
        )

    phrases, sub_version = load_substitutions() if substitute else ({}, "")

    t_seg = time.perf_counter()
    blocks = segment(text)
    timings["segment"] = (time.perf_counter() - t_seg) * 1000

    t_sub = time.perf_counter()
    out_parts: list[str] = []
    applied_map: dict[str, str] = {}
    for block in blocks:
        if block.kind is BlockKind.PROSE and substitute:
            res = apply_substitution(block.text, phrases, version=sub_version)
            out_parts.append(res.text)
            for s in res.applied:
                applied_map[s.original] = s.replacement
        else:
            out_parts.append(block.text)
    compressed = "".join(out_parts)
    timings["substitute"] = (time.perf_counter() - t_sub) * 1000

    tokens_after = counter.count(compressed)
    timings["total"] = (time.perf_counter() - t0) * 1000

    return CompressionResult(
        original=text,
        compressed=compressed,
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        token_counter=counter.name,
        ratio=(tokens_after / tokens_before) if tokens_before else 1.0,
        deleted=(),
        vetoed=(),
        substitutions=applied_map,
        protect_list_version="none",
        config_hash=config_hash,
        timings_ms=timings,
        warnings=(),
    )
