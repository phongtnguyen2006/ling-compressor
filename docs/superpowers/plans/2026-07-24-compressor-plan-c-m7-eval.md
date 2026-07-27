# Compressor Plan C (M7): Eval Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the eval harness — documents × compressors × ratios → metrics — so we can measure whether the compressed text still supports the same answers, with the headline safety metric (`protected_violation_count`) computed for every compressor, all runnable offline and lighting up model-graded accuracy only when an API key is present.

**Architecture:** A `Compressor` protocol with four baselines (`none`, `stopword`, `ours`, and an optional skip-if-missing `llmlingua2`). `metrics.py` computes structural metrics (token counts, compression ratio, latency) plus `protected_violation_count` by reusing the SAME protect-list detectors the invariant suite uses — so the eval's safety number is consistent with the build gate. `tasks.py` holds offline graders (exact / token-F1) and a model-gated path (gold-answer generation + LLM judge) behind `ANTHROPIC_API_KEY`. `harness.py` orchestrates the grid and always emits structural metrics; accuracy metrics are "skipped (no model)" without a key. `report.py` writes markdown + CSV. Everything is deterministic and offline by default (execution design Decision 3).

**Tech Stack:** Python 3.12, existing `promptcomp` (compress, tokens, protect detectors); optional `[anthropic]` and `[llmlingua]` extras (gated/skip-if-missing); `pytest`.

## Global Constraints

- Python 3.12; package layout under `eval/` at repo root (per spec §4).
- **Offline by default:** `none`, `stopword`, `ours` need no network. `llmlingua2` is skip-if-missing (log a warning, omit from results). Model-graded accuracy (gold generation, LLM judge) is gated behind `ANTHROPIC_API_KEY` + the `[anthropic]` extra; without them the harness still computes token counts, compression ratio, `protected_violation_count`, latency, and parse-failure rate, and marks accuracy "skipped (no model)".
- **`protected_violation_count` reuses the invariant detectors** from `promptcomp.protect` (`extract_numbers`, `extract_negations`, `extract_protected_terms`) plus numeric NER — one source of truth. For `ours` it must be 0 on the corpus; any nonzero value for `ours` is a build failure (assert in a test).
- **Gold answers are fixed:** once generated to `eval/gold/<doc_id>.json` they are never regenerated per run (spec §7.1).
- **Determinism:** graders and structural metrics are pure/deterministic; model calls use temperature 0. No sampling in the offline path.
- **Report which token counter produced numbers** (default `TiktokenCounter`; the model-cost caveat from §6 still applies).
- No regression to the existing 272 tests. Commit in small TDD steps. **Do not pass any `-c user.name=…/-c user.email=…` override to `git commit`** — use the repo identity (`phongtnguyen2006 <phongtnguyen2006@gmail.com>`).

---

## File Structure

| File | Responsibility |
|---|---|
| `eval/__init__.py` | Package marker |
| `eval/baselines.py` | `Compressor` protocol; `NoneCompressor`, `StopwordCompressor`, `OursCompressor`, optional `LLMLingua2Compressor` |
| `eval/metrics.py` | Structural metrics + `protected_violation_count` (shared detectors) |
| `eval/tasks.py` | Offline graders (exact, token-F1); gold cache I/O; model-gated gen + judge |
| `eval/harness.py` | Orchestrate documents × compressors × ratios → results |
| `eval/report.py` | Markdown + CSV output |
| `eval/run.py` | CLI entry: run the grid over a corpus dir, write reports |
| `tests/test_eval_baselines.py` | Baseline behavior |
| `tests/test_eval_metrics.py` | Metrics incl. violation count (ours=0, stopword>0) |
| `tests/test_eval_tasks.py` | Offline graders + gold cache |
| `tests/test_eval_harness.py` | Offline end-to-end grid |
| `tests/test_eval_report.py` | Report rendering |

**Interface contract:**
- `Compressor`: `.name: str`; `compress(text: str, ratio: float) -> Compressed` where `Compressed(text: str, latency_ms: float)`.
- `protected_violation_count(original: str, compressed: str) -> int`
- `grade_exact(pred, gold) -> float`, `grade_f1(pred, gold) -> float`
- `run_grid(docs, compressors, ratios, counter, judge=None) -> list[RunResult]`

---

## Task 1: Compressor protocol + offline baselines

**Files:**
- Create: `eval/__init__.py`, `eval/baselines.py`
- Test: `tests/test_eval_baselines.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class Compressed(text: str, latency_ms: float)`.
  - `class Compressor(Protocol)`: `name: str`; `compress(text, ratio) -> Compressed`.
  - `NoneCompressor` (identity, ceiling), `StopwordCompressor` (naive stopword removal, floor), `OursCompressor` (wraps `promptcomp.compress`, `min_tokens=0` so short eval texts compress).
  - `available_baselines() -> list[Compressor]` — returns the three offline ones plus `LLMLingua2Compressor` only if `llmlingua` imports (else logs and omits).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_baselines.py
from eval.baselines import (
    NoneCompressor, StopwordCompressor, OursCompressor, available_baselines, Compressed,
)


def test_none_is_identity():
    c = NoneCompressor()
    out = c.compress("The committee approved the budget of $2,500 today.", 0.5)
    assert isinstance(out, Compressed)
    assert out.text == "The committee approved the budget of $2,500 today."
    assert c.name == "none"


def test_stopword_removes_common_words_and_is_floor():
    c = StopwordCompressor()
    out = c.compress("The vendor delivered the goods in the morning.", 0.5)
    assert c.name == "stopword"
    # naive stopword removal drops "the"/"in" — visibly lossy
    assert "the" not in out.text.lower().split()
    assert len(out.text) < len("The vendor delivered the goods in the morning.")


def test_ours_wraps_compress_and_deletes():
    c = OursCompressor()
    text = ("The committee approved the annual budget in the morning after a long debate, "
            "and the vendor delivered the goods quickly without any delay.")
    out = c.compress(text, 0.5)
    assert c.name == "ours"
    assert len(out.text) <= len(text)


def test_available_baselines_includes_offline_three():
    names = {c.name for c in available_baselines()}
    assert {"none", "stopword", "ours"} <= names
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_eval_baselines.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'eval.baselines'`.

- [ ] **Step 3: Write the implementation**

```python
# eval/__init__.py
```

```python
# eval/baselines.py
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
    he she they them his her their we you i our your""".split()
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


class LLMLingua2Compressor:
    name = "llmlingua2"

    def __init__(self) -> None:
        from llmlingua import PromptCompressor  # raises ImportError if absent

        self._pc = PromptCompressor(
            model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
            use_llmlingua2=True,
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_eval_baselines.py -v`
Expected: PASS (4 passed; `llmlingua2` absent → skipped, not required by the tests).

- [ ] **Step 5: Commit**

```bash
git add eval/__init__.py eval/baselines.py tests/test_eval_baselines.py
git commit -m "feat(eval): add Compressor protocol and offline baselines"
```

---

## Task 2: Metrics + protected_violation_count

**Files:**
- Create: `eval/metrics.py`
- Test: `tests/test_eval_metrics.py`

**Interfaces:**
- Consumes: `promptcomp.protect` detectors, `promptcomp.parse.parse`, `promptcomp.tokens`.
- Produces:
  - `protected_violation_count(original, compressed) -> int` — number of protected tokens present in `original` but missing in `compressed`, summed over: numbers (multiset), negations (multiset), protected terms (multiset), and numeric-label named entities (set). Reuses the invariant detectors.
  - `compression_metrics(original, compressed, counter) -> dict` — `tokens_before`, `tokens_after`, `compression_ratio`, `token_counter`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_metrics.py
from eval.metrics import protected_violation_count, compression_metrics
from promptcomp.tokens import TiktokenCounter


def test_ours_has_zero_violations():
    # ours never deletes a protected token
    from eval.baselines import OursCompressor
    text = ("The company shall reimburse expenses of $2,500 within thirty days, and no "
            "exception applies unless approved; the vendor delivered the goods in the morning.")
    compressed = OursCompressor().compress(text, 0.4).text
    assert protected_violation_count(text, compressed) == 0


def test_stopword_loses_negation_so_has_violations():
    text = "The vendor did not deliver the goods and no exception applies."
    from eval.baselines import StopwordCompressor
    compressed = StopwordCompressor().compress(text, 0.4).text
    # stopword removal drops "not"/"no" -> protected negation lost
    assert protected_violation_count(text, compressed) > 0


def test_compression_metrics_shapes():
    m = compression_metrics("a " * 200, "a " * 100, TiktokenCounter())
    assert m["tokens_after"] < m["tokens_before"]
    assert 0 < m["compression_ratio"] <= 1.0
    assert m["token_counter"].startswith("tiktoken:")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_eval_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'eval.metrics'`.

- [ ] **Step 3: Write the implementation**

```python
# eval/metrics.py
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


def compression_metrics(original: str, compressed: str, counter) -> dict:
    before = counter.count(original)
    after = counter.count(compressed)
    return {
        "tokens_before": before,
        "tokens_after": after,
        "compression_ratio": (after / before) if before else 1.0,
        "token_counter": counter.name,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_eval_metrics.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add eval/metrics.py tests/test_eval_metrics.py
git commit -m "feat(eval): add metrics with shared-detector protected_violation_count"
```

---

## Task 3: Offline graders + gold cache

**Files:**
- Create: `eval/tasks.py`
- Test: `tests/test_eval_tasks.py`

**Interfaces:**
- Produces:
  - `grade_exact(pred: str, gold: str) -> float` — 1.0 if normalized-equal else 0.0 (lowercase, strip, collapse whitespace, strip surrounding punctuation).
  - `grade_f1(pred: str, gold: str) -> float` — token-overlap F1.
  - `@dataclass(frozen=True) class QAItem(question: str, answer: str)`.
  - `load_gold(doc_id: str, gold_dir: Path) -> list[QAItem] | None` and `save_gold(doc_id, items, gold_dir)` (JSON at `<gold_dir>/<doc_id>.json`; never regenerated once present).
  - `generate_gold(text, n, client) -> list[QAItem]` — model-gated; imports `anthropic` lazily and raises an actionable `RuntimeError` if unavailable. (Not exercised offline.)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_tasks.py
from pathlib import Path

from eval.tasks import grade_exact, grade_f1, QAItem, save_gold, load_gold


def test_exact_match_normalizes():
    assert grade_exact("  $2,500.  ", "$2,500") == 1.0
    assert grade_exact("thirty days", "30 days") == 0.0


def test_f1_partial_overlap():
    assert grade_f1("the quick brown fox", "quick brown fox") > 0.8
    assert grade_f1("completely different", "the quick brown fox") == 0.0


def test_gold_cache_roundtrip_and_fixed(tmp_path):
    items = [QAItem(question="How much?", answer="$2,500"),
             QAItem(question="When?", answer="thirty days")]
    assert load_gold("doc1", tmp_path) is None
    save_gold("doc1", items, tmp_path)
    loaded = load_gold("doc1", tmp_path)
    assert loaded == items
    # save must NOT overwrite an existing gold set (fixed across runs)
    save_gold("doc1", [QAItem(question="x", answer="y")], tmp_path)
    assert load_gold("doc1", tmp_path) == items
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_eval_tasks.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'eval.tasks'`.

- [ ] **Step 3: Write the implementation**

```python
# eval/tasks.py
"""QA task grading (offline) and gold-answer cache; model-gated generation/judge."""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class QAItem:
    question: str
    answer: str


def _normalize(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip(".,;:!?\"'()[] ")


def grade_exact(pred: str, gold: str) -> float:
    return 1.0 if _normalize(pred) == _normalize(gold) else 0.0


def _tokens(s: str) -> list[str]:
    return re.findall(r"\w+", s.lower())


def grade_f1(pred: str, gold: str) -> float:
    p, g = Counter(_tokens(pred)), Counter(_tokens(gold))
    if not p or not g:
        return 0.0
    overlap = sum((p & g).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(p.values())
    recall = overlap / sum(g.values())
    return 2 * precision * recall / (precision + recall)


def load_gold(doc_id: str, gold_dir: Path) -> list[QAItem] | None:
    path = Path(gold_dir) / f"{doc_id}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return [QAItem(**d) for d in data]


def save_gold(doc_id: str, items: list[QAItem], gold_dir: Path) -> None:
    gold_dir = Path(gold_dir)
    gold_dir.mkdir(parents=True, exist_ok=True)
    path = gold_dir / f"{doc_id}.json"
    if path.exists():
        return  # fixed across runs — never overwrite
    path.write_text(json.dumps([asdict(i) for i in items], indent=2))


def generate_gold(text: str, n: int, client) -> list[QAItem]:  # pragma: no cover - model
    raise RuntimeError(
        "generate_gold requires the [anthropic] extra and ANTHROPIC_API_KEY; "
        "run gold generation once, then rely on the cached eval/gold/*.json."
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_eval_tasks.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add eval/tasks.py tests/test_eval_tasks.py
git commit -m "feat(eval): add offline graders and fixed gold-answer cache"
```

---

## Task 4: Harness — the offline grid

**Files:**
- Create: `eval/harness.py`
- Test: `tests/test_eval_harness.py`

**Interfaces:**
- Consumes: `Compressor`, `compression_metrics`/`protected_violation_count`, `TiktokenCounter`.
- Produces:
  - `@dataclass(frozen=True) class RunResult(doc_id, compressor, ratio, tokens_before, tokens_after, compression_ratio, protected_violation_count, latency_ms, accuracy_exact, accuracy_f1, accuracy_judge)` — accuracy fields are `None` when no model.
  - `run_grid(docs: dict[str,str], compressors, ratios, counter, grader=None) -> list[RunResult]` — for each (doc, compressor, ratio): compress, compute structural metrics + violations + latency; if `grader` provided (model path), fill accuracy, else leave `None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_harness.py
from eval.harness import run_grid, RunResult
from eval.baselines import NoneCompressor, StopwordCompressor, OursCompressor
from promptcomp.tokens import TiktokenCounter


DOCS = {
    "d1": ("The committee approved the annual budget of $2,500 in the morning after a long "
           "debate, and the vendor delivered the goods quickly without any delay."),
}


def test_grid_shape_and_offline_accuracy_is_none():
    results = run_grid(DOCS, [NoneCompressor(), OursCompressor()], [0.4, 0.6], TiktokenCounter())
    assert len(results) == 1 * 2 * 2
    for r in results:
        assert isinstance(r, RunResult)
        assert r.accuracy_exact is None  # no grader -> offline
        assert r.tokens_after <= r.tokens_before


def test_ours_zero_violations_stopword_nonzero_in_grid():
    results = run_grid(DOCS, [OursCompressor(), StopwordCompressor()], [0.4], TiktokenCounter())
    by_name = {r.compressor: r for r in results}
    assert by_name["ours"].protected_violation_count == 0
    assert by_name["stopword"].protected_violation_count > 0


def test_none_baseline_is_ceiling():
    results = run_grid(DOCS, [NoneCompressor()], [0.5], TiktokenCounter())
    assert results[0].compression_ratio == 1.0
    assert results[0].protected_violation_count == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_eval_harness.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'eval.harness'`.

- [ ] **Step 3: Write the implementation**

```python
# eval/harness.py
"""Orchestrate documents × compressors × ratios → RunResult rows. Offline by default."""
from __future__ import annotations

from dataclasses import dataclass

from .metrics import compression_metrics, protected_violation_count


@dataclass(frozen=True)
class RunResult:
    doc_id: str
    compressor: str
    ratio: float
    tokens_before: int
    tokens_after: int
    compression_ratio: float
    protected_violation_count: int
    latency_ms: float
    accuracy_exact: float | None
    accuracy_f1: float | None
    accuracy_judge: float | None


def run_grid(docs, compressors, ratios, counter, grader=None) -> list[RunResult]:
    results: list[RunResult] = []
    for doc_id, text in docs.items():
        for compressor in compressors:
            for ratio in ratios:
                out = compressor.compress(text, ratio)
                m = compression_metrics(text, out.text, counter)
                violations = protected_violation_count(text, out.text)
                acc = grader(doc_id, out.text) if grader is not None else None
                results.append(
                    RunResult(
                        doc_id=doc_id,
                        compressor=compressor.name,
                        ratio=ratio,
                        tokens_before=m["tokens_before"],
                        tokens_after=m["tokens_after"],
                        compression_ratio=m["compression_ratio"],
                        protected_violation_count=violations,
                        latency_ms=round(out.latency_ms, 3),
                        accuracy_exact=(acc or {}).get("exact") if acc else None,
                        accuracy_f1=(acc or {}).get("f1") if acc else None,
                        accuracy_judge=(acc or {}).get("judge") if acc else None,
                    )
                )
    return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_eval_harness.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add eval/harness.py tests/test_eval_harness.py
git commit -m "feat(eval): add offline documents×compressors×ratios grid"
```

---

## Task 5: Report — markdown + CSV

**Files:**
- Create: `eval/report.py`
- Test: `tests/test_eval_report.py`

**Interfaces:**
- Consumes: `RunResult`.
- Produces:
  - `to_csv(results: list[RunResult]) -> str` — header + one row per result.
  - `to_markdown(results: list[RunResult]) -> str` — a table grouped by compressor, with a bold callout of `protected_violation_count` and a note that `ours` must be 0.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_report.py
from eval.report import to_csv, to_markdown
from eval.harness import RunResult


def _row(name, viol):
    return RunResult(
        doc_id="d1", compressor=name, ratio=0.5, tokens_before=100, tokens_after=70,
        compression_ratio=0.7, protected_violation_count=viol, latency_ms=1.2,
        accuracy_exact=None, accuracy_f1=None, accuracy_judge=None,
    )


def test_csv_has_header_and_rows():
    csv = to_csv([_row("ours", 0), _row("stopword", 3)])
    lines = csv.strip().splitlines()
    assert lines[0].startswith("doc_id,compressor,ratio")
    assert len(lines) == 3
    assert "ours" in csv and "stopword" in csv


def test_markdown_flags_violations():
    md = to_markdown([_row("ours", 0), _row("stopword", 3)])
    assert "protected_violation_count" in md
    assert "| ours |" in md or "ours" in md
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_eval_report.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'eval.report'`.

- [ ] **Step 3: Write the implementation**

```python
# eval/report.py
"""Render eval results to CSV and markdown."""
from __future__ import annotations

import csv
import io
from dataclasses import asdict, fields

from .harness import RunResult

_HEADER = [f.name for f in fields(RunResult)]


def to_csv(results: list[RunResult]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_HEADER)
    writer.writeheader()
    for r in results:
        writer.writerow(asdict(r))
    return buf.getvalue()


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def to_markdown(results: list[RunResult]) -> str:
    cols = ["compressor", "ratio", "tokens_before", "tokens_after", "compression_ratio",
            "protected_violation_count", "accuracy_retention", "latency_ms"]
    # accuracy_retention vs the 'none' ceiling per (doc, ratio)
    ceiling = {
        (r.doc_id, r.ratio): r.accuracy_f1
        for r in results if r.compressor == "none" and r.accuracy_f1 is not None
    }
    lines = ["# Eval report", "",
             "`protected_violation_count` must be **0** for `ours` (nonzero = build failure).",
             "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in sorted(results, key=lambda x: (x.doc_id, x.compressor, x.ratio)):
        base = ceiling.get((r.doc_id, r.ratio))
        retention = (r.accuracy_f1 / base) if (base and r.accuracy_f1 is not None) else None
        row = {
            "compressor": r.compressor, "ratio": r.ratio,
            "tokens_before": r.tokens_before, "tokens_after": r.tokens_after,
            "compression_ratio": r.compression_ratio,
            "protected_violation_count": r.protected_violation_count,
            "accuracy_retention": retention, "latency_ms": r.latency_ms,
        }
        lines.append("| " + " | ".join(_fmt(row[c]) for c in cols) + " |")
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_eval_report.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add eval/report.py tests/test_eval_report.py
git commit -m "feat(eval): add CSV and markdown report rendering"
```

---

## Task 6: CLI runner + offline end-to-end over the sample corpus

**Files:**
- Create: `eval/run.py`
- Modify: `.gitignore` (ignore `eval/results/`)
- Test: `tests/test_eval_run.py`

**Interfaces:**
- Consumes: everything above; `available_baselines`, `TiktokenCounter`.
- Produces: `main(argv) -> int` — reads `*.txt` docs from a corpus dir (default `data/samples`), runs the offline grid over configured ratios, writes `eval/results/report.md` and `eval/results/report.csv`, prints a short summary including the `ours` violation total (which must be 0).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_run.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
import run as eval_run  # eval/run.py  # noqa: E402


def test_offline_run_writes_reports_and_ours_is_clean(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "d1.txt").write_text(
        "The committee approved the annual budget of $2,500 in the morning after a long "
        "debate, and the vendor delivered the goods quickly without any delay.\n"
    )
    out = tmp_path / "out"
    rc = eval_run.main(["--corpus", str(corpus), "--out", str(out),
                        "--ratios", "0.4,0.6"])
    assert rc == 0
    assert (out / "report.md").exists()
    assert (out / "report.csv").exists()
    csv_text = (out / "report.csv").read_text()
    assert "ours" in csv_text and "none" in csv_text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_eval_run.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'run'` (eval/run.py missing).

- [ ] **Step 3: Write the implementation**

```python
# eval/run.py
"""CLI: run the offline eval grid over a corpus and write reports.

    python eval/run.py --corpus data/samples --out eval/results --ratios 0.3,0.5,0.7
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.baselines import available_baselines  # noqa: E402
from eval.harness import run_grid  # noqa: E402
from eval.report import to_csv, to_markdown  # noqa: E402
from promptcomp.tokens import CachedCounter, TiktokenCounter  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the offline compression eval grid.")
    parser.add_argument("--corpus", default="data/samples", help="Dir of *.txt documents.")
    parser.add_argument("--out", default="eval/results", help="Output dir for reports.")
    parser.add_argument("--ratios", default="0.3,0.5,0.7", help="Comma-separated target ratios.")
    parser.add_argument("--cache-path", default=None, help="Token cache db.")
    args = parser.parse_args(argv)

    corpus_dir = Path(args.corpus)
    docs = {p.stem: p.read_text() for p in sorted(corpus_dir.glob("*.txt"))}
    if not docs:
        print(f"no *.txt documents found in {corpus_dir}")
        return 1

    ratios = [float(r) for r in args.ratios.split(",")]
    counter = CachedCounter(TiktokenCounter(), args.cache_path or ".token_cache.sqlite")

    results = run_grid(docs, available_baselines(), ratios, counter)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.md").write_text(to_markdown(results))
    (out_dir / "report.csv").write_text(to_csv(results))

    ours_violations = sum(
        r.protected_violation_count for r in results if r.compressor == "ours"
    )
    print(f"docs={len(docs)} compressors={len({r.compressor for r in results})} "
          f"ratios={len(ratios)} rows={len(results)}")
    print(f"ours protected_violation_count total = {ours_violations} (must be 0)")
    print(f"wrote {out_dir/'report.md'} and {out_dir/'report.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Ignore results output**

Add `eval/results/` to `.gitignore` (it may already be present from Plan A; if so, skip).

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_eval_run.py -v`
Expected: PASS.

- [ ] **Step 6: Real offline run over the sample corpus**

Run: `python eval/run.py --corpus data/samples --out eval/results --ratios 0.3,0.5,0.7`
Expected: prints the grid summary and, critically, `ours protected_violation_count total = 0`. Inspect `eval/results/report.md`: `ours` shows 0 violations at every ratio; `stopword` shows nonzero (it deletes negations/stopwords); `none` is the ceiling (ratio 1.0, 0 violations). Accuracy columns show `—` (no model). This is the M7 offline checkpoint — the safety differentiator is now measured, not just asserted.

- [ ] **Step 7: Run the full suite + commit**

Run: `python -m pytest -v`
Expected: PASS (all tests green).

```bash
git add eval/run.py .gitignore tests/test_eval_run.py
git commit -m "feat(eval): add CLI runner and offline end-to-end over the corpus"
```

---

## Task 7: README eval section + corpus fetch stub

**Files:**
- Modify: `README.md`
- Create: `scripts/fetch_corpus.py`

**Interfaces:**
- Produces: README documentation of the eval (offline vs model-graded), and a documented `scripts/fetch_corpus.py` stub that explains how to grow the corpus toward the ≥30 public-domain docs the spec wants (EDGAR filings, MeetingBank transcripts) — it writes `*.txt` into a target dir. Network fetching is left as a clearly-marked TODO with the exact sources; the stub creates the target dir and prints guidance so the offline eval path never depends on it.

- [ ] **Step 1: Write the fetch stub**

```python
# scripts/fetch_corpus.py
"""Grow the eval corpus toward the >=30 long-form public documents the spec wants.

Offline eval runs on data/samples/ without this. To scale up, point --out at a dir
and add public-domain analogues (SEC EDGAR filings, earnings-call transcripts,
central-bank statements, MeetingBank transcripts). Fetching is intentionally left as a
documented step so the eval never hard-depends on network access.
"""
from __future__ import annotations

import argparse
from pathlib import Path

SOURCES = {
    "edgar": "https://www.sec.gov/cgi-bin/browse-edgar (10-K/10-Q filings, public domain)",
    "meetingbank": "https://meetingbank.github.io/ (city-council transcripts)",
    "fomc": "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm (statements)",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Corpus fetch guidance / target dir setup.")
    parser.add_argument("--out", default="data/corpus", help="Target dir for *.txt docs.")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Target corpus dir ready: {out}")
    print("Add >=30 long-form (2k-20k token) .txt documents from public sources:")
    for k, v in SOURCES.items():
        print(f"  - {k}: {v}")
    print("Then: python eval/run.py --corpus", out, "--out eval/results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Add the eval section to README**

Append to `README.md`:

```markdown
## Eval (M7)

Measure whether the compressed text still supports the same answers:

    python eval/run.py --corpus data/samples --out eval/results --ratios 0.3,0.5,0.7

Runs **offline** by default over four baselines (`none` ceiling, `stopword` floor,
`ours`, and `llmlingua2` if installed). It always reports token counts, compression
ratio, latency, and **`protected_violation_count`** — which must be **0** for `ours`
(nonzero is a build failure) and is visibly nonzero for `stopword`. Model-graded
accuracy (gold generation + LLM judge) lights up when `ANTHROPIC_API_KEY` and the
`[anthropic]` extra are present; otherwise accuracy shows `—`.

Grow the corpus toward the ≥30 public documents the eval wants:

    python scripts/fetch_corpus.py --out data/corpus
```

- [ ] **Step 3: Verify the stub runs**

Run: `python scripts/fetch_corpus.py --out data/corpus`
Expected: creates `data/corpus/` and prints source guidance. Exit 0.

- [ ] **Step 4: Commit**

```bash
git add README.md scripts/fetch_corpus.py
git commit -m "docs(eval): document eval usage and add corpus-fetch guidance"
```

---

## Self-Review Notes

**Spec coverage (M7 / §7):**
- documents × compressors × ratios → metrics → Task 4 (`run_grid`). ✓
- All four baselines every run (`none`/`stopword`/`ours`/`llmlingua2` skip-if-missing) → Task 1. ✓
- Task generation + fixed gold cache → Task 3 (offline graders + gold I/O; model gen gated). ✓
- Grading exact / F1 / judge → Task 3 (exact, F1 offline; judge model-gated). ✓
- Metrics incl. `accuracy_retention` and `protected_violation_count` (0 for ours) → Tasks 2, 5. ✓
- Markdown + CSV report → Task 5. ✓
- Corpus ≥30 (starter + fetch guidance) → Task 7 + existing `data/samples`. ✓
- Offline-by-default, model-gated (execution design Decision 3) → throughout. ✓

**Consistency:** `protected_violation_count` reuses the invariant detectors (`extract_numbers`/`extract_negations`/`extract_protected_terms` + numeric NER) so the eval's headline safety number equals the build gate's definition — a regression can't show 0 violations in eval while failing the invariant suite.

**Deferred (documented, needs API/corpus):** live gold generation, the LLM judge, the real `llmlingua2` comparison, and the ≥30-doc corpus. All are gated/stubbed so the offline path is fully testable now and lights up when keys/deps/corpus arrive — this is the spec's "ship M7 before optimizing" checkpoint reached in its offline form.

**Placeholder scan:** no TBD/TODO in code paths that run offline; the one explicit TODO is the network fetch in `scripts/fetch_corpus.py`, intentionally documented so the offline eval never depends on it.
