from __future__ import annotations

import hashlib
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class TokenCounter(Protocol):
    @property
    def name(self) -> str: ...
    def count(self, text: str) -> int: ...


class TiktokenCounter:
    def __init__(self, encoding: str = "o200k_base") -> None:
        import tiktoken

        self._encoding_name = encoding
        self._enc = tiktoken.get_encoding(encoding)

    @property
    def name(self) -> str:
        return f"tiktoken:{self._encoding_name}"

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._enc.encode(text))


class AnthropicCounter:
    """Exact Claude token counts. Gated behind the [anthropic] extra + API key."""

    def __init__(self, model: str = "claude-opus-4-8") -> None:
        self._model = model

    @property
    def name(self) -> str:
        return f"anthropic:{self._model}"

    def count(self, text: str) -> int:
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover - depends on env
            raise RuntimeError(
                "AnthropicCounter requires the 'anthropic' package. "
                "Install with: pip install 'promptcomp[anthropic]' and set ANTHROPIC_API_KEY."
            ) from e
        try:
            client = anthropic.Anthropic()
            resp = client.messages.count_tokens(
                model=self._model,
                messages=[{"role": "user", "content": text}],
            )
            return resp.input_tokens
        except Exception as e:  # pragma: no cover - network/auth dependent
            raise RuntimeError(
                f"AnthropicCounter failed ({type(e).__name__}: {e}). Ensure the "
                "anthropic package is installed and ANTHROPIC_API_KEY is set."
            ) from e


class CachedCounter:
    """Wraps a counter, caching results by sha256(text)+inner.name in sqlite."""

    def __init__(self, inner: TokenCounter, db_path: str | Path) -> None:
        self._inner = inner
        self._db_path = str(db_path)
        self._init_db()

    @property
    def name(self) -> str:
        return self._inner.name

    def _init_db(self) -> None:
        with closing(sqlite3.connect(self._db_path)) as conn, conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS token_counts "
                "(key TEXT PRIMARY KEY, tokens INTEGER NOT NULL)"
            )

    def _key(self, text: str) -> str:
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"{h}:{self._inner.name}"

    def count(self, text: str) -> int:
        key = self._key(text)
        with closing(sqlite3.connect(self._db_path)) as conn, conn:
            row = conn.execute(
                "SELECT tokens FROM token_counts WHERE key = ?", (key,)
            ).fetchone()
            if row is not None:
                return int(row[0])
            tokens = self._inner.count(text)
            conn.execute(
                "INSERT OR REPLACE INTO token_counts (key, tokens) VALUES (?, ?)",
                (key, tokens),
            )
            return tokens
