"""Explicitly provisioned tokenizer assets for the offline lexical experiment."""
from __future__ import annotations

from threading import RLock
from unittest.mock import patch

from ..tokens import TiktokenCounter

_INITIALIZATION_LOCK = RLock()


def local_counter(encoding: str = "o200k_base") -> TiktokenCounter:
    """Load cached tiktoken assets or fail with an explicit provisioning command.

    tiktoken normally fetches a missing cache entry lazily. During initialization
    its underlying uncached reader is blocked, while cached, hash-validated
    reads still work. The guard is restored on success and failure. Initialize
    during application setup and reuse the resulting counter; the lock serializes
    this helper's calls, not arbitrary third-party tiktoken initializations.
    """
    import tiktoken.load

    if encoding not in {"o200k_base", "cl100k_base"}:
        raise ValueError("The offline lexical pilot supports o200k_base and cl100k_base")

    def unavailable(path):
        raise RuntimeError(
            f"Tokenizer asset is not cached locally ({path}). Explicitly initialize it first with: "
            f"python -c \"import tiktoken; tiktoken.get_encoding('{encoding}')\""
        )

    with _INITIALIZATION_LOCK, patch.object(tiktoken.load, "read_file", unavailable):
        return TiktokenCounter(encoding)
