# Compressor Plan A (M1–M3): Skeleton, Tokens, Segment, Substitute Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the dependency-light foundation of the linguistic prompt compressor — core types, dual token counting with caching, Stage 0 segmentation (PROSE vs PASSTHROUGH), Stage 4 lossless substitution, and an inspection CLI — producing a working, lossless compressor with zero ML dependencies.

**Architecture:** A `src/promptcomp/` package (src-layout) holding frozen dataclasses for the domain model, a `TokenCounter` protocol with tiktoken (default) and Anthropic (gated) backends behind a sqlite cache, a regex/heuristic segmenter that splits text into offset-preserving `Block`s, a dictionary-driven substitution pass, and a `scripts/inspect.py` CLI that renders a colored diff. No spaCy, no torch — those arrive in Plan B (M4–M6). `pipeline.compress()` in this plan wires only Stage 0 + Stage 4 (segment then substitute); parse/protect/rank land in Plan B.

**Tech Stack:** Python 3.12, `tiktoken`, `pyyaml`, `pytest`, `hypothesis`; optional `anthropic` and `llmlingua` extras (not exercised in Plan A). src-layout via `pyproject.toml` (setuptools or hatchling).

## Global Constraints

- Python floor: **3.12** (project uses 3.12.3).
- Package name: **`promptcomp`**, src-layout at `src/promptcomp/`.
- **Determinism is a requirement:** same input + same config + same protect-list version → byte-identical output. No sampling, no wall-clock in outputs except `timings_ms`.
- **Offsets are sacred:** every `Block` carries `start`/`end` char offsets into the original text; recombining all blocks in order must reproduce the input exactly (byte-identical).
- **Passthrough is never modified:** PASSTHROUGH blocks pass through byte-identical.
- **Do not use tiktoken as a proxy for Claude costs.** Every token count records which counter produced it.
- **Substitution is lossless/reversible:** phrase-shortening is one-way-safe; entity abbreviation is bijective and recorded in `CompressionResult.substitutions`.
- Heavy deps (`anthropic`, `llmlingua`, `torch`, `spacy`) are **optional extras**, imported lazily, never required for core install or the Plan A test suite.
- All new code committed in small TDD steps.

---

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | Package metadata, src-layout, core deps + `[anthropic]`/`[llmlingua]` extras |
| `src/promptcomp/__init__.py` | Package marker + public re-exports |
| `src/promptcomp/types.py` | Frozen dataclasses: `Span`, `Block`, `Deletion`, `Veto`, `Substitution`, `CompressionResult`; `BlockKind` enum |
| `src/promptcomp/tokens.py` | `TokenCounter` protocol, `TiktokenCounter`, `AnthropicCounter` (gated), sqlite cache |
| `src/promptcomp/segment.py` | Stage 0: `segment(text) -> list[Block]` |
| `src/promptcomp/substitute.py` | Stage 4: `substitute(text) -> SubstituteResult` |
| `src/promptcomp/pipeline.py` | `compress()` wiring Stage 0 + Stage 4 (Plan A subset) |
| `data/substitutions.yaml` | Phrase-shortening + entity-abbreviation dictionaries |
| `scripts/inspect.py` | CLI: colored diff, token counts, per-stage timing |
| `tests/test_types.py` | Dataclass invariants |
| `tests/test_tokens.py` | Counter behavior + caching |
| `tests/test_segment.py` | Passthrough integrity + classification |
| `tests/test_substitute.py` | Lossless/reversible substitution |
| `tests/test_pipeline.py` | End-to-end Plan A compress() (segment+substitute) |

**Interface contract carried into Plan B** (do not break these signatures):
- `segment(text: str) -> list[Block]` where `Block(kind: BlockKind, text: str, start: int, end: int)`.
- `compress(text, *, target_ratio=0.6, substitute=True, min_tokens=500, ...) -> CompressionResult` — Plan B extends the body; the signature and return type are fixed here.
- `CompressionResult` field names are fixed here and consumed by the eval harness (Plan C).

---

## Task 1: Package scaffolding + pyproject

**Files:**
- Create: `pyproject.toml`
- Create: `src/promptcomp/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/test_smoke.py`

**Interfaces:**
- Consumes: nothing.
- Produces: an installable `promptcomp` package importable as `import promptcomp`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_smoke.py
def test_package_imports():
    import promptcomp
    assert promptcomp.__version__ == "0.1.0"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_smoke.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'promptcomp'`

- [ ] **Step 3: Write pyproject.toml**

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "promptcomp"
version = "0.1.0"
description = "Local, deterministic, non-generative linguistic prompt compressor"
requires-python = ">=3.12"
dependencies = [
    "tiktoken>=0.7",
    "pyyaml>=6",
]

[project.optional-dependencies]
anthropic = ["anthropic>=0.40"]
llmlingua = ["llmlingua>=0.2", "torch>=2"]
dev = ["pytest>=8", "hypothesis>=6"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 4: Write the package init**

```python
# src/promptcomp/__init__.py
"""Local, deterministic, non-generative linguistic prompt compressor."""

__version__ = "0.1.0"
```

```python
# tests/__init__.py
```

- [ ] **Step 5: Install editable and run the test**

Run: `python -m pip install -e ".[dev]" && python -m pytest tests/test_smoke.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/promptcomp/__init__.py tests/__init__.py tests/test_smoke.py
git commit -m "chore: scaffold promptcomp package with pyproject and extras"
```

---

## Task 2: Core types

**Files:**
- Create: `src/promptcomp/types.py`
- Test: `tests/test_types.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `class BlockKind(str, Enum)`: `PROSE = "prose"`, `PASSTHROUGH = "passthrough"`.
  - `@dataclass(frozen=True) class Span(start: int, end: int)` with `def text(self, source: str) -> str` returning `source[start:end]`.
  - `@dataclass(frozen=True) class Block(kind: BlockKind, text: str, start: int, end: int)`.
  - `@dataclass(frozen=True) class Deletion(span: Span, text: str, deprel: str, score: float, reason: str)`.
  - `@dataclass(frozen=True) class Veto(span: Span, text: str, protect_class: str)`.
  - `@dataclass(frozen=True) class Substitution(original: str, replacement: str, reversible: bool)`.
  - `@dataclass(frozen=True) class CompressionResult` with fields: `original: str`, `compressed: str`, `tokens_before: int`, `tokens_after: int`, `token_counter: str`, `ratio: float`, `deleted: tuple[Deletion, ...]`, `vetoed: tuple[Veto, ...]`, `substitutions: dict[str, str]`, `protect_list_version: str`, `config_hash: str`, `timings_ms: dict[str, float]`, `warnings: tuple[str, ...]`.

Note: mutable containers in a frozen dataclass use `tuple`/immutable defaults; `substitutions` and `timings_ms` are dicts populated at construction (the frozen guarantee is shallow, which is acceptable — results are not mutated after construction). Add `token_counter` and `warnings` beyond the spec's §5 list so the counter is always attributable (Decision 2) and parse/segment warnings surface (Decision 4).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_types.py
import dataclasses
import pytest
from promptcomp.types import (
    BlockKind, Span, Block, Deletion, Veto, Substitution, CompressionResult,
)


def test_span_text_slices_source():
    span = Span(start=6, end=11)
    assert span.text("hello world") == "world"


def test_block_kind_values():
    assert BlockKind.PROSE.value == "prose"
    assert BlockKind.PASSTHROUGH.value == "passthrough"


def test_result_is_frozen():
    r = CompressionResult(
        original="a", compressed="a", tokens_before=1, tokens_after=1,
        token_counter="tiktoken:o200k_base", ratio=1.0, deleted=(), vetoed=(),
        substitutions={}, protect_list_version="none", config_hash="x",
        timings_ms={}, warnings=(),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.compressed = "b"


def test_block_frozen_and_offsets():
    b = Block(kind=BlockKind.PROSE, text="hi", start=0, end=2)
    assert b.text == "hi" and b.start == 0 and b.end == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_types.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'promptcomp.types'`

- [ ] **Step 3: Write the implementation**

```python
# src/promptcomp/types.py
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class BlockKind(str, Enum):
    PROSE = "prose"
    PASSTHROUGH = "passthrough"


@dataclass(frozen=True)
class Span:
    start: int
    end: int

    def text(self, source: str) -> str:
        return source[self.start : self.end]


@dataclass(frozen=True)
class Block:
    kind: BlockKind
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class Deletion:
    span: Span
    text: str
    deprel: str
    score: float
    reason: str


@dataclass(frozen=True)
class Veto:
    span: Span
    text: str
    protect_class: str


@dataclass(frozen=True)
class Substitution:
    original: str
    replacement: str
    reversible: bool


@dataclass(frozen=True)
class CompressionResult:
    original: str
    compressed: str
    tokens_before: int
    tokens_after: int
    token_counter: str
    ratio: float
    deleted: tuple[Deletion, ...]
    vetoed: tuple[Veto, ...]
    substitutions: dict[str, str]
    protect_list_version: str
    config_hash: str
    timings_ms: dict[str, float]
    warnings: tuple[str, ...] = field(default_factory=tuple)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_types.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/promptcomp/types.py tests/test_types.py
git commit -m "feat: add core domain types (Span, Block, CompressionResult, ...)"
```

---

## Task 3: Token counting with sqlite cache

**Files:**
- Create: `src/promptcomp/tokens.py`
- Test: `tests/test_tokens.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `class TokenCounter(Protocol)`: `def count(self, text: str) -> int`; property `name: str`.
  - `class TiktokenCounter(encoding: str = "o200k_base")` — `.name == f"tiktoken:{encoding}"`.
  - `class AnthropicCounter(model: str = "claude-opus-4-8")` — lazily imports `anthropic`; raises `RuntimeError` with an actionable message if the package or `ANTHROPIC_API_KEY` is missing; `.name == f"anthropic:{model}"`.
  - `class CachedCounter(inner: TokenCounter, db_path: str | Path)` — wraps any counter, caching by `sha256(text)+inner.name` in sqlite; `.name == inner.name`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tokens.py
import pytest
from promptcomp.tokens import TiktokenCounter, CachedCounter, AnthropicCounter


def test_tiktoken_counts_are_positive_and_named():
    c = TiktokenCounter()
    assert c.name == "tiktoken:o200k_base"
    assert c.count("hello world, this is a test") > 3


def test_tiktoken_empty_is_zero():
    assert TiktokenCounter().count("") == 0


class _CountingCounter:
    """Counts how many times the inner counter is actually invoked."""
    name = "tiktoken:o200k_base"

    def __init__(self):
        self.calls = 0

    def count(self, text: str) -> int:
        self.calls += 1
        return len(text.split())


def test_cache_avoids_recount(tmp_path):
    inner = _CountingCounter()
    cached = CachedCounter(inner, db_path=tmp_path / "cache.sqlite")
    assert cached.count("a b c") == 3
    assert cached.count("a b c") == 3
    assert inner.calls == 1  # second call served from cache


def test_cache_persists_across_instances(tmp_path):
    db = tmp_path / "cache.sqlite"
    inner1 = _CountingCounter()
    CachedCounter(inner1, db_path=db).count("x y")
    inner2 = _CountingCounter()
    assert CachedCounter(inner2, db_path=db).count("x y") == 2
    assert inner2.calls == 0  # served from disk cache written by inner1


def test_anthropic_counter_missing_dep_is_actionable():
    # Without the anthropic package / key, constructing-then-counting must raise
    # a clear RuntimeError rather than an opaque ImportError.
    c = AnthropicCounter()
    with pytest.raises(RuntimeError) as exc:
        c.count("hello")
    msg = str(exc.value).lower()
    assert "anthropic" in msg
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_tokens.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'promptcomp.tokens'`

Note: `test_anthropic_counter_missing_dep_is_actionable` assumes the `anthropic` package is not installed (Plan A does not install the extra). If it happens to be installed and a key is set, this test is environment-dependent; keep the extra uninstalled for Plan A.

- [ ] **Step 3: Write the implementation**

```python
# src/promptcomp/tokens.py
from __future__ import annotations

import hashlib
import sqlite3
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
                f"AnthropicCounter failed ({e}). Ensure ANTHROPIC_API_KEY is set "
                "and the anthropic package is installed."
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
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS token_counts "
                "(key TEXT PRIMARY KEY, tokens INTEGER NOT NULL)"
            )

    def _key(self, text: str) -> str:
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"{h}:{self._inner.name}"

    def count(self, text: str) -> int:
        key = self._key(text)
        with sqlite3.connect(self._db_path) as conn:
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_tokens.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/promptcomp/tokens.py tests/test_tokens.py
git commit -m "feat: add token counters (tiktoken, anthropic-gated) with sqlite cache"
```

---

## Task 4: Stage 0 segmentation — passthrough integrity

**Files:**
- Create: `src/promptcomp/segment.py`
- Test: `tests/test_segment.py`

**Interfaces:**
- Consumes: `Block`, `BlockKind` from `types`.
- Produces: `def segment(text: str) -> list[Block]`. Blocks are contiguous and offset-preserving: `blocks[i].end == blocks[i+1].start`, `blocks[0].start == 0`, `blocks[-1].end == len(text)`, and `"".join(b.text for b in blocks) == text`.

Classification (spec §3 Stage 0), line-oriented over the raw text:
- Fenced code blocks (```` ``` ```` … ```` ``` ````) → PASSTHROUGH (whole fence incl. delimiters).
- Markdown table rows (lines containing `|` with a `---`/`:--` separator row) → PASSTHROUGH.
- Lines that are a URL, a file path, or a stack-trace frame → PASSTHROUGH.
- Lines with >30% digit characters or >40% non-alpha (non-space) characters → PASSTHROUGH.
- Everything else → PROSE.

Adjacent lines of the same kind coalesce into one Block. Newlines belong to the block of the line they terminate. This is deterministic and regex/heuristic only — no ML.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_segment.py
from hypothesis import given, strategies as st
from promptcomp.segment import segment
from promptcomp.types import BlockKind


def _reassembles(text: str) -> bool:
    blocks = segment(text)
    if not blocks:
        return text == ""
    if blocks[0].start != 0 or blocks[-1].end != len(text):
        return False
    for a, b in zip(blocks, blocks[1:]):
        if a.end != b.start:
            return False
    return "".join(b.text for b in blocks) == text


def test_reassembly_exact_simple():
    assert _reassembles("The quick brown fox.\nAnother sentence here.\n")


def test_fenced_code_is_passthrough():
    text = "Intro prose line.\n```\ncode = 1\n```\nOutro prose.\n"
    blocks = segment(text)
    kinds = [(b.kind, b.text) for b in blocks]
    assert any(k is BlockKind.PASSTHROUGH and "code = 1" in t for k, t in kinds)
    assert any(k is BlockKind.PROSE and "Intro prose" in t for k, t in kinds)
    assert _reassembles(text)


def test_digit_heavy_line_is_passthrough():
    text = "Normal explanatory sentence about the account.\n1234 5678 9012 3456\n"
    blocks = segment(text)
    assert any(
        b.kind is BlockKind.PASSTHROUGH and "1234 5678" in b.text for b in blocks
    )


def test_url_line_is_passthrough():
    text = "See the reference below for details.\nhttps://example.com/a/b/c\n"
    blocks = segment(text)
    assert any(
        b.kind is BlockKind.PASSTHROUGH and "example.com" in b.text for b in blocks
    )


def test_markdown_table_is_passthrough():
    text = (
        "Here is the breakdown.\n"
        "| Col A | Col B |\n"
        "| --- | --- |\n"
        "| 1 | 2 |\n"
        "Closing remark.\n"
    )
    blocks = segment(text)
    assert any(b.kind is BlockKind.PASSTHROUGH and "Col A" in b.text for b in blocks)


@given(st.text())
def test_reassembly_never_loses_bytes(text):
    assert _reassembles(text)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_segment.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'promptcomp.segment'`

- [ ] **Step 3: Write the implementation**

```python
# src/promptcomp/segment.py
from __future__ import annotations

import re

from .types import Block, BlockKind

_URL_RE = re.compile(r"^\s*https?://\S+\s*$")
_PATH_RE = re.compile(r"^\s*(/[^\s/]+){2,}/?\s*$")
_STACK_RE = re.compile(r'^\s*(File \"|at\s+\S+\(|Traceback|\s+at\s)')
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")


def _split_lines_keepends(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def _is_digit_heavy(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    digits = sum(c.isdigit() for c in stripped)
    return digits / len(stripped) > 0.30


def _is_nonalpha_heavy(line: str) -> bool:
    non_space = [c for c in line if not c.isspace()]
    if not non_space:
        return False
    nonalpha = sum(not c.isalpha() for c in non_space)
    return nonalpha / len(non_space) > 0.40


def _looks_like_table_row(line: str, lines: list[str], idx: int) -> bool:
    if "|" not in line:
        return False
    # a table needs a separator row somewhere adjacent
    for j in (idx - 1, idx + 1):
        if 0 <= j < len(lines) and _TABLE_SEP_RE.match(lines[j]):
            return True
    return bool(_TABLE_SEP_RE.match(line))


def _classify_line(line: str, lines: list[str], idx: int) -> BlockKind:
    if _URL_RE.match(line) or _PATH_RE.match(line) or _STACK_RE.match(line):
        return BlockKind.PASSTHROUGH
    if _looks_like_table_row(line, lines, idx):
        return BlockKind.PASSTHROUGH
    if _is_digit_heavy(line) or _is_nonalpha_heavy(line):
        return BlockKind.PASSTHROUGH
    return BlockKind.PROSE


def segment(text: str) -> list[Block]:
    if text == "":
        return []

    lines = _split_lines_keepends(text)
    # First pass: classify each line, honoring fenced-code state.
    kinds: list[BlockKind] = []
    in_fence = False
    for idx, line in enumerate(lines):
        is_fence_delim = line.lstrip().startswith("```")
        if in_fence:
            kinds.append(BlockKind.PASSTHROUGH)
            if is_fence_delim:
                in_fence = False
            continue
        if is_fence_delim:
            in_fence = True
            kinds.append(BlockKind.PASSTHROUGH)
            continue
        kinds.append(_classify_line(line, lines, idx))

    # Second pass: coalesce adjacent same-kind lines into offset-preserving blocks.
    blocks: list[Block] = []
    pos = 0
    run_start = 0
    run_kind = kinds[0]
    run_text: list[str] = []
    for line, kind in zip(lines, kinds):
        if kind != run_kind and run_text:
            joined = "".join(run_text)
            blocks.append(Block(run_kind, joined, run_start, run_start + len(joined)))
            run_start = run_start + len(joined)
            run_kind = kind
            run_text = []
        run_text.append(line)
        pos += len(line)
    if run_text:
        joined = "".join(run_text)
        blocks.append(Block(run_kind, joined, run_start, run_start + len(joined)))
    return blocks
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_segment.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/promptcomp/segment.py tests/test_segment.py
git commit -m "feat: add Stage 0 segmentation with passthrough integrity"
```

---

## Task 5: Substitution data file

**Files:**
- Create: `data/substitutions.yaml`
- Test: `tests/test_substitute.py` (data-loading portion only in this task)

**Interfaces:**
- Consumes: nothing (a data file).
- Produces: a YAML document with two top-level keys:
  - `phrases:` — map of long form → short form (one-way-safe phrase shortening).
  - `version:` — a string stamped into audit output later.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_substitute.py
import yaml
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data" / "substitutions.yaml"


def test_substitutions_file_loads_and_has_phrases():
    doc = yaml.safe_load(DATA.read_text())
    assert "version" in doc
    assert isinstance(doc["phrases"], dict)
    assert doc["phrases"]["in order to"] == "to"
    assert doc["phrases"]["due to the fact that"] == "because"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_substitute.py::test_substitutions_file_loads_and_has_phrases -v`
Expected: FAIL — `FileNotFoundError` on `data/substitutions.yaml`

- [ ] **Step 3: Write the data file**

```yaml
# data/substitutions.yaml
version: "sub-2026-07-23.1"

# One-way-safe phrase shortening. Keys matched case-insensitively as whole
# words; longer keys are applied before shorter overlapping ones.
phrases:
  "in order to": "to"
  "due to the fact that": "because"
  "at this point in time": "now"
  "for the purpose of": "for"
  "in the event that": "if"
  "with regard to": "about"
  "in the near future": "soon"
  "a large number of": "many"
  "at the present time": "now"
  "in spite of the fact that": "although"
  "on a daily basis": "daily"
  "in the majority of cases": "usually"
  "has the ability to": "can"
  "is able to": "can"
  "prior to": "before"
  "subsequent to": "after"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_substitute.py::test_substitutions_file_loads_and_has_phrases -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add data/substitutions.yaml tests/test_substitute.py
git commit -m "feat: add phrase-shortening substitution dictionary"
```

---

## Task 6: Stage 4 substitution pass

**Files:**
- Create: `src/promptcomp/substitute.py`
- Modify: `tests/test_substitute.py` (add behavior tests)

**Interfaces:**
- Consumes: `Substitution` from `types`; `data/substitutions.yaml`.
- Produces:
  - `@dataclass(frozen=True) class SubstituteResult(text: str, applied: tuple[Substitution, ...], version: str)`.
  - `def load_substitutions(path: str | Path | None = None) -> tuple[dict[str, str], str]` returning `(phrases, version)`; default path resolves to the repo `data/substitutions.yaml`.
  - `def substitute(text: str, phrases: dict[str, str] | None = None, *, version: str = "") -> SubstituteResult`. Applies phrase shortening case-insensitively on whole-word boundaries, longest key first. **PROSE only** — callers pass prose text; substitute does not itself segment. Records each applied replacement (with `reversible=False` for phrase shortening). Deterministic.

Only phrase-shortening (safe, one-way) is implemented in Plan A. Entity abbreviation (bijective) is deferred — it depends on defined-term extraction from Stage 1/protect (Plan B); note this in a module docstring.

- [ ] **Step 1: Write the failing test (append to tests/test_substitute.py)**

```python
# tests/test_substitute.py (append)
from promptcomp.substitute import substitute, load_substitutions, SubstituteResult


def test_substitute_shortens_known_phrase():
    res = substitute("We did this in order to win.", {"in order to": "to"})
    assert res.text == "We did this to win."
    assert any(s.original == "in order to" for s in res.applied)


def test_substitute_is_case_insensitive_preserves_nothing_extra():
    res = substitute("In order to proceed, wait.", {"in order to": "to"})
    assert res.text == "to proceed, wait."


def test_substitute_longest_match_first():
    phrases = {"due to": "because of", "due to the fact that": "because"}
    res = substitute("He left due to the fact that it rained.", phrases)
    assert res.text == "He left because it rained."


def test_substitute_no_match_is_identity():
    res = substitute("Nothing to change here.", {"in order to": "to"})
    assert res.text == "Nothing to change here."
    assert res.applied == ()


def test_substitute_does_not_break_words():
    # "for" inside "form" must not be touched by a "for" -> "4" style rule.
    res = substitute("Fill the form for me.", {"for": "4"})
    assert "form" in res.text  # 'form' intact
    assert res.text == "Fill the form 4 me."


def test_load_substitutions_default_path():
    phrases, version = load_substitutions()
    assert phrases["in order to"] == "to"
    assert version.startswith("sub-")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_substitute.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'promptcomp.substitute'`

- [ ] **Step 3: Write the implementation**

```python
# src/promptcomp/substitute.py
"""Stage 4: lossless, dictionary-driven substitution.

Only phrase-shortening (safe, one-way) is implemented here. Entity abbreviation
(bijective, reversible) depends on defined-term extraction from Stage 1/protect
and lands in Plan B (M4-M6).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from .types import Substitution

_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "substitutions.yaml"


@dataclass(frozen=True)
class SubstituteResult:
    text: str
    applied: tuple[Substitution, ...]
    version: str


def load_substitutions(path: str | Path | None = None) -> tuple[dict[str, str], str]:
    p = Path(path) if path is not None else _DEFAULT_PATH
    doc = yaml.safe_load(p.read_text())
    return dict(doc.get("phrases", {})), str(doc.get("version", ""))


def substitute(
    text: str,
    phrases: dict[str, str] | None = None,
    *,
    version: str = "",
) -> SubstituteResult:
    if phrases is None:
        phrases, version = load_substitutions()
    if not phrases or not text:
        return SubstituteResult(text=text, applied=(), version=version)

    # Longest keys first so multi-word phrases win over their prefixes.
    keys = sorted(phrases, key=len, reverse=True)
    applied: list[Substitution] = []
    result = text
    for key in keys:
        pattern = re.compile(r"\b" + re.escape(key) + r"\b", re.IGNORECASE)
        if pattern.search(result):
            replacement = phrases[key]
            result, n = pattern.subn(replacement, result)
            if n:
                applied.append(
                    Substitution(original=key, replacement=replacement, reversible=False)
                )
    return SubstituteResult(text=result, applied=tuple(applied), version=version)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_substitute.py -v`
Expected: PASS (7 passed)

Note: `test_substitute_is_case_insensitive_preserves_nothing_extra` expects capitalized "In order to" at sentence start to become lowercase "to" — this is acceptable for Plan A (phrase shortening does not preserve sentence-initial casing). If later deemed wrong, casing preservation is a Plan B refinement.

- [ ] **Step 5: Commit**

```bash
git add src/promptcomp/substitute.py tests/test_substitute.py
git commit -m "feat: add Stage 4 phrase-shortening substitution pass"
```

---

## Task 7: Plan A pipeline — compress() wiring segment + substitute

**Files:**
- Create: `src/promptcomp/pipeline.py`
- Modify: `src/promptcomp/__init__.py` (re-export `compress`)
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `segment`, `substitute`/`load_substitutions`, `TiktokenCounter`, `CachedCounter`, `CompressionResult`.
- Produces: `def compress(text, *, target_ratio=0.6, scorer=None, substitute=True, min_tokens=500, max_deletion_fraction=0.5, counter=None, cache_path=None) -> CompressionResult`.

Plan A behavior (no parse/protect/rank yet):
- Size gate: if `counter.count(text) < min_tokens`, return unchanged (ratio 1.0), still populating token counts and `config_hash`.
- Otherwise: `segment(text)`; for each PROSE block, apply `substitute` (if enabled); PASSTHROUGH blocks pass through byte-identical; recombine in original order.
- `deleted` and `vetoed` are empty tuples in Plan A (deletion arrives in Plan B). `substitutions` is the merged applied map. `config_hash` is `sha256` of the sorted config kwargs. `protect_list_version` is `"none"` for Plan A.
- The default counter is `CachedCounter(TiktokenCounter(), cache_path or ".token_cache.sqlite")`.

Note the parameter name shadow: the boolean kwarg `substitute` shadows the imported `substitute` function inside `compress`. Import the function under an alias (`from .substitute import substitute as apply_substitution`) to avoid the collision.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pipeline.py
from promptcomp.pipeline import compress


def test_short_text_passes_through_unchanged():
    text = "This is a short instruction under the size gate."
    result = compress(text, min_tokens=500)
    assert result.compressed == text
    assert result.ratio == 1.0
    assert result.deleted == ()
    assert result.token_counter.startswith("tiktoken:")


def test_prose_substitution_applies_above_gate(tmp_path):
    # Force the gate open with min_tokens=0 so short fixtures still compress.
    text = "We proceeded in order to finish, due to the fact that time was short."
    result = compress(text, min_tokens=0, cache_path=tmp_path / "c.sqlite")
    assert "in order to" not in result.compressed
    assert "due to the fact that" not in result.compressed
    assert result.tokens_after <= result.tokens_before
    assert "in order to" in result.substitutions


def test_passthrough_block_is_byte_identical(tmp_path):
    text = "Explanation prose here in order to demonstrate.\n```\nx = in order to\n```\n"
    result = compress(text, min_tokens=0, cache_path=tmp_path / "c.sqlite")
    # The code fence content must be untouched even though it contains the phrase.
    assert "x = in order to" in result.compressed


def test_substitute_disabled_is_identity(tmp_path):
    text = "We did this in order to win the contract."
    result = compress(
        text, substitute=False, min_tokens=0, cache_path=tmp_path / "c.sqlite"
    )
    assert result.compressed == text


def test_determinism_same_input_same_output(tmp_path):
    text = "We proceeded in order to finish the report on time today."
    r1 = compress(text, min_tokens=0, cache_path=tmp_path / "c1.sqlite")
    r2 = compress(text, min_tokens=0, cache_path=tmp_path / "c2.sqlite")
    assert r1.compressed == r2.compressed
    assert r1.config_hash == r2.config_hash
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'promptcomp.pipeline'`

- [ ] **Step 3: Write the implementation**

```python
# src/promptcomp/pipeline.py
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
```

- [ ] **Step 4: Re-export from package init**

```python
# src/promptcomp/__init__.py
"""Local, deterministic, non-generative linguistic prompt compressor."""

from .pipeline import compress
from .types import CompressionResult

__version__ = "0.1.0"

__all__ = ["compress", "CompressionResult", "__version__"]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: PASS (5 passed)

- [ ] **Step 6: Commit**

```bash
git add src/promptcomp/pipeline.py src/promptcomp/__init__.py tests/test_pipeline.py
git commit -m "feat: add Plan A compress() pipeline (segment + substitute)"
```

---

## Task 8: Inspection CLI

**Files:**
- Create: `scripts/inspect.py`
- Test: `tests/test_inspect.py`

**Interfaces:**
- Consumes: `compress` from `promptcomp`.
- Produces: a CLI runnable as `python scripts/inspect.py --file <path> --ratio 0.6` and `--stdin`, `--out result.json`. Prints token counts before/after (with counter name), per-stage timing, applied substitutions, and a warnings section. Deletions coloring is a no-op in Plan A (nothing is deleted yet) but the code path exists.

Keep it dependency-free (stdlib `argparse`, `json`, `sys`). Provide a `main(argv)` function returning an int exit code so it is unit-testable without a subprocess.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_inspect.py
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import inspect as inspect_cli  # scripts/inspect.py  # noqa: E402


def test_main_writes_json_out(tmp_path, capsys):
    doc = tmp_path / "doc.txt"
    doc.write_text("We proceeded in order to finish the work quickly today.\n")
    out = tmp_path / "result.json"
    rc = inspect_cli.main(
        ["--file", str(doc), "--ratio", "0.6", "--min-tokens", "0",
         "--out", str(out), "--cache-path", str(tmp_path / "c.sqlite")]
    )
    assert rc == 0
    data = json.loads(out.read_text())
    assert data["tokens_before"] >= data["tokens_after"]
    assert "in order to" not in data["compressed"]
    assert data["token_counter"].startswith("tiktoken:")


def test_main_stdin(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.stdin",
        type("S", (), {"read": staticmethod(lambda: "Nothing special to compress.")})(),
    )
    rc = inspect_cli.main(
        ["--stdin", "--min-tokens", "0", "--cache-path", str(tmp_path / "c.sqlite")]
    )
    assert rc == 0
    captured = capsys.readouterr()
    assert "tokens" in captured.out.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_inspect.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'inspect'` resolving to `scripts/inspect.py` (file does not exist yet; the stdlib `inspect` may shadow — see Step 3 note).

- [ ] **Step 3: Write the implementation**

Note: naming the file `inspect.py` shadows the stdlib `inspect` module for this test because `scripts/` is prepended to `sys.path`. That is acceptable here (the script does not import stdlib `inspect`). The spec names it `scripts/inspect.py`; keep that name.

```python
# scripts/inspect.py
"""Inspection harness: paste/feed text, see the compression result.

Usage:
    python scripts/inspect.py --file data/samples/memo1.txt --ratio 0.6
    python scripts/inspect.py --stdin --out result.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

# Make the src-layout package importable when run as a loose script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from promptcomp import compress  # noqa: E402


def _result_to_dict(result) -> dict:
    d = dataclasses.asdict(result)
    return d


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect compression output.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", type=str, help="Path to a text file to compress.")
    src.add_argument("--stdin", action="store_true", help="Read text from stdin.")
    parser.add_argument("--ratio", type=float, default=0.6, help="Target ratio.")
    parser.add_argument("--min-tokens", type=int, default=500, help="Size gate.")
    parser.add_argument("--no-substitute", action="store_true", help="Disable Stage 4.")
    parser.add_argument("--out", type=str, default=None, help="Write JSON result here.")
    parser.add_argument("--cache-path", type=str, default=None, help="Token cache db.")
    args = parser.parse_args(argv)

    if args.stdin:
        text = sys.stdin.read()
    else:
        text = Path(args.file).read_text()

    result = compress(
        text,
        target_ratio=args.ratio,
        substitute=not args.no_substitute,
        min_tokens=args.min_tokens,
        cache_path=args.cache_path,
    )

    print(f"token counter : {result.token_counter}")
    print(f"tokens before : {result.tokens_before}")
    print(f"tokens after  : {result.tokens_after}")
    print(f"ratio         : {result.ratio:.3f}")
    print(f"timings (ms)  : {result.timings_ms}")
    if result.substitutions:
        print("substitutions :")
        for k, v in result.substitutions.items():
            print(f"    {k!r} -> {v!r}")
    if result.warnings:
        print("warnings      :")
        for w in result.warnings:
            print(f"    - {w}")

    if args.out:
        Path(args.out).write_text(json.dumps(_result_to_dict(result), indent=2))
        print(f"wrote         : {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_inspect.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Run the full Plan A suite**

Run: `python -m pytest -v`
Expected: PASS (all tests green across all test files)

- [ ] **Step 6: Commit**

```bash
git add scripts/inspect.py tests/test_inspect.py
git commit -m "feat: add inspection CLI for compression output"
```

---

## Task 9: README + sample document

**Files:**
- Create: `README.md`
- Create: `data/samples/memo1.txt`
- Test: manual verification (documented command)

**Interfaces:**
- Consumes: everything above.
- Produces: a README documenting install, the token-counter caveat, and a runnable inspect command; one long-form sample doc (>500 tiktoken tokens) so the size gate opens without `--min-tokens 0`.

- [ ] **Step 1: Create a long-form sample document**

Create `data/samples/memo1.txt` with a realistic long-form prose memo of **at least ~700 words** (so it clears the 500-token gate). Use public-domain-style, non-sensitive content: e.g. a synthetic internal-policy memo describing an expense-reimbursement policy with amounts, dates, conditions, and obligations (these exercise the future protect-list). Include one fenced code block and one short URL line so segmentation has passthrough content to catch. Do not use real confidential text.

- [ ] **Step 2: Verify the sample clears the gate**

Run: `python scripts/inspect.py --file data/samples/memo1.txt --ratio 0.6`
Expected: prints token counts with `tokens before` > 500, applies at least one substitution, and does not error. The fenced code block content appears unchanged.

- [ ] **Step 3: Write the README**

```markdown
# promptcomp — Linguistic Prompt Compressor (Pass 1)

Local, deterministic, non-generative compressor that shrinks long-form prose token
counts while never deleting protected tokens. See `COMPRESSOR_SPEC.md` for the full
design and `docs/superpowers/specs/` for execution decisions.

**Status:** Plan A (M1–M3) — segmentation + lossless substitution. Parsing,
protect-list, and ranking (deletion) arrive in Plan B.

## Install

    python -m pip install -e ".[dev]"

Optional extras (not required for Plan A):

    python -m pip install -e ".[anthropic]"   # exact Claude token counts
    python -m pip install -e ".[llmlingua]"   # eval llmlingua2 baseline

## Token counting caveat

Do **not** use tiktoken as a proxy for Claude cost. Claude Opus 4.7+, Sonnet 5,
Fable 5, and Mythos 5 use a newer tokenizer producing ~30% more tokens for the same
text than earlier models. Use `AnthropicCounter` for any cost/budget decision; every
reported count records which counter produced it.

## Try it

    python scripts/inspect.py --file data/samples/memo1.txt --ratio 0.6

## Test

    python -m pytest -v
```

- [ ] **Step 4: Commit**

```bash
git add README.md data/samples/memo1.txt
git commit -m "docs: add README and long-form sample document"
```

---

## Self-Review Notes

**Spec coverage (M1–M3 subset):**
- M1 skeleton + tokens → Tasks 1–3 (`types.py`, `tokens.py` both counters, cache). ✓
- M2 segmentation + passthrough integrity tests → Task 4. ✓
- M3 substitution pass + standalone measurement → Tasks 5–6; standalone effect visible via `inspect.py` and `test_pipeline`. ✓
- `scripts/inspect.py` (§5) with `--stdin`/`--out` → Task 8. ✓
- Token-counter attribution + caveat (§6) → Task 3 + README. ✓
- Determinism (§11.4) → `test_pipeline::test_determinism...`, `config_hash`. ✓

**Deferred to Plan B (intentional, documented):** dependency parsing, protect-list veto, ranking/budget/deletion, entity-abbreviation substitution, defined-term extraction, the full invariant suite (`test_protect_invariants.py`), grammaticality re-parse. `compress()` signature already carries `scorer`, `target_ratio`, `max_deletion_fraction` so Plan B extends the body without changing callers.

**Interface stability:** `segment() -> list[Block]`, `compress(...) -> CompressionResult`, and all `CompressionResult` field names are fixed here and must not change in Plan B/C.

**Placeholder scan:** no TBD/TODO; every code step contains complete code. The one prose-authored artifact is `data/samples/memo1.txt` (Task 9), which is content, not code.
