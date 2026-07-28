"""Compressors under evaluation: the ceiling (none), floor (stopword), ours, and
an optional llmlingua2 baseline (skip-if-missing)."""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from promptcomp import compress

log = logging.getLogger(__name__)

# A small, deterministic English stopword set for the naive floor baseline.
_STOPWORDS = frozenset(
    """a an the of to in on at for and or but if then else when while is are was were be been
    being this that these those with without by from as into over under above below it its it's
    he she they them his her their we you i our your not no""".split()
)


@dataclass(frozen=True)
class Compressed:
    text: str
    latency_ms: float


@runtime_checkable
class Compressor(Protocol):
    @property
    def name(self) -> str: ...
    def compress(self, text: str, ratio: float) -> Compressed: ...


class NoneCompressor:
    name = "none"

    def compress(self, text: str, ratio: float) -> Compressed:
        return Compressed(text=text, latency_ms=0.0)


class StopwordCompressor:
    name = "stopword"

    def compress(self, text: str, ratio: float) -> Compressed:
        t0 = time.perf_counter()
        tokens = re.findall(r"\w+|[^\w\s]|\s+", text)
        kept = [tok for tok in tokens if tok.strip().lower() not in _STOPWORDS]
        out = "".join(kept)
        out = re.sub(r"\s{2,}", " ", out)
        return Compressed(text=out, latency_ms=(time.perf_counter() - t0) * 1000)


class OursCompressor:
    name = "ours"

    def compress(self, text: str, ratio: float) -> Compressed:
        t0 = time.perf_counter()
        result = compress(text, target_ratio=ratio, min_tokens=0)
        return Compressed(text=result.compressed, latency_ms=(time.perf_counter() - t0) * 1000)


def _best_device() -> str:
    """Pick an available torch device. LLMLingua defaults to cuda, which is absent
    on Apple Silicon / CPU-only hosts and raises at construction time."""
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class LLMLingua2Compressor:
    name = "llmlingua2"

    def __init__(self, device_map: str | None = None) -> None:
        from llmlingua import PromptCompressor  # raises ImportError if absent

        if device_map is None:
            device_map = _best_device()
        self._pc = PromptCompressor(
            model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
            use_llmlingua2=True,
            device_map=device_map,
        )

    def compress(self, text: str, ratio: float) -> Compressed:
        t0 = time.perf_counter()
        out = self._pc.compress_prompt(text, rate=ratio, force_tokens=["\n", ".", "?", "!"])
        return Compressed(
            text=out["compressed_prompt"], latency_ms=(time.perf_counter() - t0) * 1000
        )


def available_baselines() -> list[Compressor]:
    baselines: list[Compressor] = [NoneCompressor(), StopwordCompressor(), OursCompressor()]
    try:
        baselines.append(LLMLingua2Compressor())
    except Exception as e:  # ImportError or model download failure
        log.warning("llmlingua2 baseline unavailable, skipping: %s", e)
    return baselines
