# Compressor Plan B3 (M6): Heuristic Scorer + Budget + Deletion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the full extractive pipeline (parse → protect-veto → rank → delete → substitute) into `compress()` so it actually deletes cheapest low-information subtrees down to a token budget, and prove with the invariant suite that no protected token is ever deleted and every output sentence stays grammatical.

**Architecture:** A `HeuristicScorer` (behind a `Scorer` protocol) assigns each veto-surviving candidate an information-cost score (lower = cheaper to delete). `delete.py` filters candidates to contiguous whole subtrees, greedily selects non-overlapping cheapest candidates until a per-block token budget is met (capped by `max_deletion_fraction`), removes their char ranges, and normalizes whitespace seams. `pipeline.compress()` orchestrates all stages per PROSE block, then applies Stage 4 substitution to the surviving prose, recombining with byte-identical PASSTHROUGH blocks. The invariant suite (the product) reuses the shared extractors from `protect.py` so protection holds by construction.

**Tech Stack:** Python 3.12, spaCy `en_core_web_sm`, existing `protect`/`parse`/`segment`/`substitute`/`tokens`; `pytest` + `hypothesis`.

## Global Constraints

- Python 3.12; package `promptcomp`, src-layout.
- **Only veto SURVIVORS are ever deletable.** The pipeline runs `veto()` and passes only survivors to scoring/selection. A protected candidate can never be selected.
- **Whole contiguous subtrees only.** Before scoring, filter survivors to candidates whose char range corresponds EXACTLY to their subtree: `span = doc.char_span(start, end)` is not `None` AND `len(span) == candidate.n_tokens`. This drops non-projective / non-contiguous candidates (the B1 carryover risk) so deletion never removes an intervening non-subtree token.
- **Determinism (hard requirement):** same input + same config + same protect-list version → byte-identical output. Sort by `(score, char_start)`; no sampling. `config_hash` now includes the scorer's identity.
- **Budget cap:** never delete more than `max_deletion_fraction` (default 0.5) of a block's tokens.
- **Size gate unchanged:** below `min_tokens` (default 500), return input unchanged.
- **Passthrough integrity:** PASSTHROUGH blocks byte-identical in output. Deletion + substitution apply to PROSE only.
- **Invariants use shared extractors:** the invariant suite imports `extract_numbers`/`extract_negations` from `protect.py` — the same functions the veto uses — so agreement is guaranteed.
- **Stamp audit fields:** `CompressionResult.deleted`, `.vetoed`, `.protect_list_version` are now populated for real.
- No regression to the existing 75 tests. Commit in small TDD steps. **Do not pass any `-c user.name=…/-c user.email=…` override to `git commit`** — use the repo identity (`phongtnguyen2006 <phongtnguyen2006@gmail.com>`).

---

## File Structure

| File | Responsibility |
|---|---|
| `src/promptcomp/scoring/__init__.py` | Package marker |
| `src/promptcomp/scoring/base.py` | `Scorer` protocol |
| `src/promptcomp/scoring/heuristic.py` | `HeuristicScorer` (deterministic info-cost) |
| `src/promptcomp/delete.py` | `is_contiguous_subtree`, `select_deletions`, `apply_deletions` |
| `src/promptcomp/pipeline.py` | `compress()` — full pipeline wiring |
| `scripts/inspect.py` | `--show-candidates` now also shows real DELETED candidates |
| `tests/test_scoring.py` | Scorer determinism + ordering |
| `tests/test_delete.py` | Contiguity guard, seam normalization, budget selection |
| `tests/test_pipeline.py` | End-to-end deletion behavior (extended) |
| `tests/test_protect_invariants.py` | THE critical suite |
| `tests/test_grammaticality.py` | Output sentences keep a ROOT |
| `tests/test_determinism.py` | Cross-call byte-identical output |

**Interface contract:**
- `Scorer.score(doc, candidates: list[Candidate]) -> list[float]` (lower = cheaper to delete); `HeuristicScorer` has `.name: str`.
- `is_contiguous_subtree(doc, cand: Candidate) -> bool`
- `select_deletions(scored: list[tuple[Candidate, float]], budget_tokens: int, max_fraction: float, total_tokens: int) -> list[tuple[Candidate, float]]`
- `apply_deletions(text: str, selected: list[tuple[Candidate, float]], base_offset: int = 0) -> tuple[str, list[Deletion]]`

---

## Task 1: Scorer protocol + HeuristicScorer

**Files:**
- Create: `src/promptcomp/scoring/__init__.py`
- Create: `src/promptcomp/scoring/base.py`
- Create: `src/promptcomp/scoring/heuristic.py`
- Test: `tests/test_scoring.py`

**Interfaces:**
- Consumes: `Candidate` from `types`; a spaCy `Doc`.
- Produces: `Scorer` protocol; `HeuristicScorer` with `.name` and `score(doc, candidates) -> list[float]` (lower score = cheaper/less information to lose = deleted first).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scoring.py
from promptcomp.parse import parse, enumerate_candidates
from promptcomp.scoring.heuristic import HeuristicScorer


def _scored(text):
    doc = parse(text)
    cands = enumerate_candidates(doc)
    scores = HeuristicScorer().score(doc, cands)
    return cands, scores


def test_scores_align_with_candidates():
    cands, scores = _scored("The committee approved the annual budget in the morning.")
    assert len(scores) == len(cands)
    assert all(isinstance(s, float) for s in scores)


def test_scorer_is_deterministic():
    doc = parse("The committee quickly approved the annual budget in the morning.")
    cands = enumerate_candidates(doc)
    s1 = HeuristicScorer().score(doc, cands)
    s2 = HeuristicScorer().score(doc, cands)
    assert s1 == s2


def test_adverb_cheaper_than_adjective():
    # a bare discourse/adverbial adjunct should be cheaper to delete than an
    # adjectival modifier (which carries content).
    doc = parse("The remarkably thorough committee approved the budget quietly.")
    cands = enumerate_candidates(doc)
    scores = dict(zip((c.text for c in cands), HeuristicScorer().score(doc, cands)))
    # "quietly" (advmod) should score lower (cheaper) than "remarkably thorough" (amod subtree)
    advmods = [v for k, v in scores.items() if k == "quietly"]
    amods = [v for k, v in scores.items() if "thorough" in k]
    assert advmods and amods
    assert min(advmods) < max(amods)


def test_scorer_has_name():
    assert HeuristicScorer().name == "heuristic"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_scoring.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'promptcomp.scoring'`.

- [ ] **Step 3: Write the protocol**

```python
# src/promptcomp/scoring/__init__.py
```

```python
# src/promptcomp/scoring/base.py
from __future__ import annotations

from typing import Protocol

from ..types import Candidate


class Scorer(Protocol):
    @property
    def name(self) -> str: ...
    def score(self, doc, candidates: list[Candidate]) -> list[float]: ...
```

- [ ] **Step 4: Write the HeuristicScorer**

```python
# src/promptcomp/scoring/heuristic.py
"""Deterministic, CPU-only information-cost scorer.

Lower score => cheaper to delete (less information lost) => deleted first.
Combines content-word density, average content-word length (a rarity proxy),
and a per-deprel prior (adjuncts cheap, adjectives/appositives dearer).
"""
from __future__ import annotations

from ..types import Candidate

# Per-deprel prior: higher => more likely to carry meaning => keep (dearer).
_DEPREL_PRIOR = {
    "discourse": 0.2,
    "advmod": 0.3,
    "prep": 0.4,
    "npadvmod": 0.4,
    "advcl": 0.5,
    "parataxis": 0.6,
    "appos": 0.6,
    "acl": 0.7,
    "relcl": 0.7,
    "nmod": 0.8,
    "amod": 1.0,
}


class HeuristicScorer:
    @property
    def name(self) -> str:
        return "heuristic"

    def score(self, doc, candidates: list[Candidate]) -> list[float]:
        scores: list[float] = []
        for cand in candidates:
            toks = [
                t
                for t in doc
                if t.idx >= cand.char_start and (t.idx + len(t.text)) <= cand.char_end
            ]
            content = [t for t in toks if not (t.is_stop or t.is_punct or t.is_space)]
            n = max(1, len(toks))
            density = len(content) / n
            avg_len = (sum(len(t.text) for t in content) / len(content)) if content else 0.0
            prior = _DEPREL_PRIOR.get(cand.deprel, 0.5)
            scores.append(round(prior * (0.5 + density) * (1.0 + avg_len / 10.0), 6))
        return scores
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_scoring.py -v`
Expected: PASS. If `test_adverb_cheaper_than_adjective` fails, print the candidate texts + scores and confirm the deprels; the priors, not the test, are what to adjust.

- [ ] **Step 6: Commit**

```bash
git add src/promptcomp/scoring tests/test_scoring.py
git commit -m "feat: add Scorer protocol and deterministic HeuristicScorer"
```

---

## Task 2: Deletion mechanics — contiguity guard + apply

**Files:**
- Create: `src/promptcomp/delete.py`
- Test: `tests/test_delete.py`

**Interfaces:**
- Consumes: `Candidate`, `Deletion`, `Span` from `types`; a spaCy `Doc`.
- Produces:
  - `is_contiguous_subtree(doc, cand) -> bool` — `True` iff `doc.char_span(start,end)` is not `None` and `len(span) == cand.n_tokens`.
  - `apply_deletions(text, selected, base_offset=0) -> tuple[str, list[Deletion]]` — removes each selected candidate's char range from `text`, normalizes whitespace seams, and records a `Deletion` per removed span (span offsets shifted by `base_offset`). `selected` is `list[tuple[Candidate, float]]` (candidate, score). Selected ranges must be non-overlapping (guaranteed by Task 3).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_delete.py
from promptcomp.parse import parse, enumerate_candidates
from promptcomp.delete import is_contiguous_subtree, apply_deletions
from promptcomp.types import Candidate


def test_contiguous_subtree_true_for_projective_pp():
    doc = parse("The committee approved the budget in the morning.")
    cands = enumerate_candidates(doc)
    pp = next(c for c in cands if "in the morning" in c.text)
    assert is_contiguous_subtree(doc, pp) is True


def test_apply_deletions_removes_range_and_normalizes_seam():
    text = "The committee approved the budget in the morning."
    doc = parse(text)
    cands = enumerate_candidates(doc)
    pp = next(c for c in cands if "in the morning" in c.text)
    out, deletions = apply_deletions(text, [(pp, 1.0)])
    assert "in the morning" not in out
    assert "  " not in out                 # no double space at the seam
    assert out == "The committee approved the budget."   # clean, grammatical remnant
    assert len(deletions) == 1
    assert deletions[0].text == "in the morning"
    assert deletions[0].deprel == pp.deprel


def test_apply_deletions_empty_is_identity():
    text = "Nothing to delete here."
    out, deletions = apply_deletions(text, [])
    assert out == text
    assert deletions == []


def test_apply_deletions_base_offset_shifts_spans():
    text = "The committee approved the budget in the morning."
    doc = parse(text)
    cands = enumerate_candidates(doc)
    pp = next(c for c in cands if "in the morning" in c.text)
    _, deletions = apply_deletions(text, [(pp, 1.0)], base_offset=100)
    assert deletions[0].span.start == pp.char_start + 100
    assert deletions[0].span.end == pp.char_end + 100
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_delete.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'promptcomp.delete'`.

- [ ] **Step 3: Write the implementation**

```python
# src/promptcomp/delete.py
"""Stage 3 deletion mechanics: contiguity guard, range removal, seam cleanup."""
from __future__ import annotations

import re

from .types import Candidate, Deletion, Span

# Whitespace-seam normalizers applied after removing spans.
_MULTISPACE = re.compile(r"[ \t]{2,}")
_SPACE_BEFORE_PUNCT = re.compile(r"[ \t]+([,.;:!?)\]])")
_SPACE_AFTER_OPEN = re.compile(r"([(\[])[ \t]+")
_SPACE_BEFORE_NL = re.compile(r"[ \t]+(\n)")
_SPACE_AFTER_NL = re.compile(r"(\n)[ \t]+")
_DOUBLE_COMMA = re.compile(r",\s*,")
_SPACE_RUN = None  # reserved


def is_contiguous_subtree(doc, cand: Candidate) -> bool:
    """True iff the candidate's char range is exactly its dependency subtree."""
    span = doc.char_span(cand.char_start, cand.char_end)
    return span is not None and len(span) == cand.n_tokens


def _normalize_seams(text: str) -> str:
    text = _DOUBLE_COMMA.sub(",", text)
    text = _MULTISPACE.sub(" ", text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _SPACE_AFTER_OPEN.sub(r"\1", text)
    text = _SPACE_BEFORE_NL.sub(r"\1", text)
    text = _SPACE_AFTER_NL.sub(r"\1", text)
    text = _MULTISPACE.sub(" ", text)
    return text


def apply_deletions(
    text: str,
    selected: list[tuple[Candidate, float]],
    base_offset: int = 0,
) -> tuple[str, list[Deletion]]:
    if not selected:
        return text, []
    ordered = sorted(selected, key=lambda cs: cs[0].char_start)
    out: list[str] = []
    deletions: list[Deletion] = []
    pos = 0
    for cand, sc in ordered:
        out.append(text[pos : cand.char_start])
        deletions.append(
            Deletion(
                span=Span(cand.char_start + base_offset, cand.char_end + base_offset),
                text=text[cand.char_start : cand.char_end],
                deprel=cand.deprel,
                score=sc,
                reason="budget",
            )
        )
        pos = cand.char_end
    out.append(text[pos:])
    return _normalize_seams("".join(out)), deletions
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_delete.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/promptcomp/delete.py tests/test_delete.py
git commit -m "feat: add deletion mechanics with contiguity guard and seam cleanup"
```

---

## Task 3: Budget selection

**Files:**
- Modify: `src/promptcomp/delete.py`
- Test: `tests/test_delete.py`

**Interfaces:**
- Consumes: `Candidate`.
- Produces: `select_deletions(scored, budget_tokens, max_fraction, total_tokens) -> list[tuple[Candidate, float]]` — `scored` is `list[tuple[Candidate, float]]`; greedily picks cheapest (lowest score) non-overlapping candidates until the sum of their `n_tokens` reaches `budget_tokens`, never exceeding `floor(max_fraction * total_tokens)` tokens removed. Deterministic tie-break `(score, char_start)`.

- [ ] **Step 1: Write the failing test (append to tests/test_delete.py)**

```python
# tests/test_delete.py  (append)
from promptcomp.delete import select_deletions
from promptcomp.types import Candidate


def _c(start, end, n):
    return Candidate(deprel="advmod", char_start=start, char_end=end,
                     head_text="x", text="x" * (end - start), n_tokens=n)


def test_select_picks_cheapest_first_until_budget():
    a = _c(0, 5, 1)    # cheap
    b = _c(10, 20, 3)  # dearer position, more tokens
    scored = [(a, 0.1), (b, 0.9)]
    sel = select_deletions(scored, budget_tokens=1, max_fraction=1.0, total_tokens=10)
    assert [c.char_start for c, _ in sel] == [0]  # only the cheapest needed to meet budget


def test_select_respects_max_fraction_cap():
    a = _c(0, 5, 4)
    b = _c(10, 20, 4)
    scored = [(a, 0.1), (b, 0.2)]
    # cap = floor(0.5 * 10) = 5 tokens; a=4 fits, adding b (4 more -> 8) exceeds cap
    sel = select_deletions(scored, budget_tokens=99, max_fraction=0.5, total_tokens=10)
    assert sum(c.n_tokens for c, _ in sel) <= 5
    assert [c.char_start for c, _ in sel] == [0]


def test_select_skips_overlapping_candidates():
    a = _c(0, 20, 3)
    b = _c(5, 10, 1)   # nested inside a
    scored = [(b, 0.1), (a, 0.2)]  # b cheaper, picked first; a overlaps -> skipped
    sel = select_deletions(scored, budget_tokens=99, max_fraction=1.0, total_tokens=100)
    starts = sorted(c.char_start for c, _ in sel)
    assert starts == [5]


def test_select_is_deterministic():
    a = _c(0, 5, 1)
    b = _c(10, 15, 1)
    scored = [(a, 0.5), (b, 0.5)]  # tie -> break by char_start
    sel = select_deletions(scored, budget_tokens=1, max_fraction=1.0, total_tokens=10)
    assert [c.char_start for c, _ in sel] == [0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_delete.py -v`
Expected: FAIL — `ImportError: cannot import name 'select_deletions'`.

- [ ] **Step 3: Write the implementation (append to delete.py)**

```python
# src/promptcomp/delete.py  (append)
def _overlaps(a: Candidate, b: Candidate) -> bool:
    return not (a.char_end <= b.char_start or b.char_end <= a.char_start)


def select_deletions(
    scored: list[tuple[Candidate, float]],
    budget_tokens: int,
    max_fraction: float,
    total_tokens: int,
) -> list[tuple[Candidate, float]]:
    cap = int(max_fraction * total_tokens)
    order = sorted(scored, key=lambda cs: (cs[1], cs[0].char_start))
    selected: list[tuple[Candidate, float]] = []
    removed = 0
    for cand, sc in order:
        if removed >= budget_tokens:
            break
        if removed + cand.n_tokens > cap:
            continue
        if any(_overlaps(cand, chosen) for chosen, _ in selected):
            continue
        selected.append((cand, sc))
        removed += cand.n_tokens
    return selected
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_delete.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/promptcomp/delete.py tests/test_delete.py
git commit -m "feat: add greedy non-overlapping budget selection for deletions"
```

---

## Task 4: Wire compress() — full pipeline

**Files:**
- Modify: `src/promptcomp/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `segment`, `parse`/`enumerate_candidates`, `veto`/`load_protect_list`/`load_defined_terms`/`extract_defined_terms`, `HeuristicScorer`, `is_contiguous_subtree`/`select_deletions`/`apply_deletions`, `substitute`/`load_substitutions`, counters, `CompressionResult`.
- Produces: `compress()` (same signature as Plan A) now doing parse → veto → filter-contiguous → score → select → apply-deletions → substitute per PROSE block, populating `deleted`, `vetoed`, `protect_list_version`. The `scorer` kwarg defaults to `HeuristicScorer()`. `config_hash` now includes `scorer.name`.

Per-PROSE-block flow (deletion budget is per block):
1. `doc = parse(block.text)`; `cands = enumerate_candidates(doc)`.
2. `survivors, block_vetoes = veto(doc, cands, protect_list, defined)`.
3. `deletable = [c for c in survivors if is_contiguous_subtree(doc, c)]`.
4. `scores = scorer.score(doc, deletable)`; `scored = list(zip(deletable, scores))`.
5. `total = len(doc)`; `budget = int((1 - target_ratio) * total)`; `selected = select_deletions(scored, budget, max_deletion_fraction, total)`.
6. `new_text, dels = apply_deletions(block.text, selected, base_offset=block.start)`.
7. If `substitute`: `new_text = apply_substitution(new_text, phrases, version=sub_version).text` and merge its applied map.
8. Accumulate `new_text` into output parts; extend `deleted` with `dels`; extend `vetoed` with block_vetoes shifted by `block.start`.
- PASSTHROUGH blocks: append byte-identical; no parse/veto/delete.
- Recombine in order. Stamp `protect_list_version = protect_list.version`.

- [ ] **Step 1: Write the failing test (append to tests/test_pipeline.py)**

```python
# tests/test_pipeline.py  (append)
def test_deletes_clean_adjunct_below_ratio(tmp_path):
    text = (
        "The committee approved the annual budget in the morning after a long debate. "
        "The vendor delivered the goods quickly."
    )
    result = compress(text, target_ratio=0.6, min_tokens=0,
                      cache_path=tmp_path / "c.sqlite")
    assert result.tokens_after < result.tokens_before
    assert len(result.deleted) >= 1
    # a deleted span must be a real adjunct, not a core argument
    assert all(d.deprel not in {"nsubj", "dobj", "ROOT"} for d in result.deleted)


def test_protected_amount_is_never_deleted(tmp_path):
    text = (
        "The company shall reimburse expenses of $2,500 within thirty days, "
        "and the vendor delivered the goods in the morning after a long delay."
    )
    result = compress(text, target_ratio=0.5, min_tokens=0,
                      cache_path=tmp_path / "c.sqlite")
    assert "$2,500" in result.compressed          # money survives
    assert "shall" in result.compressed           # modal survives
    assert any(v.protect_class in {"number", "modal"} for v in result.vetoed)


def test_passthrough_still_byte_identical_with_deletion(tmp_path):
    text = (
        "The committee approved the budget in the morning after long debate.\n"
        "```\ncode = in order to stay\n```\n"
    )
    result = compress(text, target_ratio=0.6, min_tokens=0,
                      cache_path=tmp_path / "c.sqlite")
    assert "code = in order to stay" in result.compressed  # fence untouched


def test_result_stamps_protect_version(tmp_path):
    result = compress("The committee approved the budget in the morning.",
                      min_tokens=0, cache_path=tmp_path / "c.sqlite")
    assert result.protect_list_version.startswith("protect-")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: FAIL — deletion not wired; `result.deleted` empty and `protect_list_version == "none"`.

- [ ] **Step 3: Rewrite pipeline.py**

```python
# src/promptcomp/pipeline.py
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

    cfg = dict(
        target_ratio=target_ratio,
        substitute=substitute,
        min_tokens=min_tokens,
        max_deletion_fraction=max_deletion_fraction,
        scorer=scorer.name,
    )
    config_hash = _config_hash(**cfg)

    tokens_before = counter.count(text)
    protect_list = load_protect_list()

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
    phrases, sub_version = load_substitutions() if substitute else ({}, "")

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: PASS. If a deletion test fails because nothing deletes (budget 0 on a tiny block), confirm `target_ratio` leaves a non-zero budget for the fixture; the fixtures above are sized to delete at least one adjunct.

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest -v`
Expected: PASS — existing tests still green (short-text gate, substitution, passthrough).

- [ ] **Step 6: Commit**

```bash
git add src/promptcomp/pipeline.py tests/test_pipeline.py
git commit -m "feat: wire full extractive pipeline into compress()"
```

---

## Task 5: The critical invariant suite

**Files:**
- Create: `tests/test_protect_invariants.py`
- Modify: `data/samples/` (add two short protected-content docs)
- Test: itself.

**Interfaces:**
- Consumes: `compress`; `extract_numbers`/`extract_negations` from `protect` (shared source of truth).
- Produces: the build-gating invariant tests.

- [ ] **Step 1: Add two small corpus documents**

Create `data/samples/policy_snip.txt` (~3–6 sentences, must exceed a few hundred chars) mixing protected content and deletable adjuncts, e.g.:

```
The Company shall reimburse eligible expenses of up to $5,000 per quarter, provided that receipts are submitted within thirty days. Employees may not claim more than 15% above the standard rate, and no exception applies unless approved in writing by the regional director. In the ordinary course of business, the finance team, which operates from the central office, processes each request carefully and without undue delay.
```

Create `data/samples/notice_snip.txt`:

```
Acme Corporation ("the Company") must deliver the goods no later than March 3, 2026, and shall not exceed the agreed budget of $2.5 million under any circumstances. The vendor, acting reasonably and in good faith, will provide at least two progress reports before the final deadline. Neither party may assign this agreement without prior written consent.
```

- [ ] **Step 2: Write the invariant tests**

```python
# tests/test_protect_invariants.py
from collections import Counter
from pathlib import Path

import pytest
from hypothesis import given, settings, strategies as st

from promptcomp import compress
from promptcomp.protect import extract_numbers, extract_negations
from promptcomp.segment import segment
from promptcomp.types import BlockKind

_SAMPLES = Path(__file__).resolve().parents[1] / "data" / "samples"
CORPUS = [p.read_text() for p in sorted(_SAMPLES.glob("*.txt"))]

# Curated sentences that stress each protected class.
CURATED = [
    "The Company shall reimburse expenses of $2,500 within thirty days after approval.",
    "Employees may not claim more than 15% above the standard rate in any month.",
    "No exception applies unless approved in writing by the regional director beforehand.",
    "Acme Corporation must deliver the goods no later than March 3, 2026, without fail.",
    "The vendor, acting in good faith, will provide at least two reports before the deadline.",
]


def _numbers_multiset(s):
    return Counter(extract_numbers(s))


@pytest.mark.parametrize("doc", CORPUS + CURATED)
@pytest.mark.parametrize("ratio", [0.3, 0.5, 0.7])
def test_numbers_multiset_preserved(doc, ratio):
    result = compress(doc, target_ratio=ratio, min_tokens=0)
    assert _numbers_multiset(result.original) == _numbers_multiset(result.compressed)


@pytest.mark.parametrize("doc", CORPUS + CURATED)
@pytest.mark.parametrize("ratio", [0.3, 0.5, 0.7])
def test_negation_count_non_decreasing(doc, ratio):
    result = compress(doc, target_ratio=ratio, min_tokens=0)
    before = len(extract_negations(result.original))
    after = len(extract_negations(result.compressed))
    assert after >= before


@pytest.mark.parametrize("doc", CORPUS + CURATED)
def test_no_vetoed_class_token_deleted(doc):
    # Every negation and number present originally must still be present.
    result = compress(doc, target_ratio=0.4, min_tokens=0)
    assert set(extract_numbers(result.original)) <= set(extract_numbers(result.compressed))
    assert Counter(extract_negations(result.compressed)) >= Counter(extract_negations(result.original))


@pytest.mark.parametrize("doc", CORPUS + CURATED)
def test_passthrough_blocks_byte_identical(doc):
    result = compress(doc, target_ratio=0.4, min_tokens=0)
    for block in segment(result.original):
        if block.kind is BlockKind.PASSTHROUGH:
            assert block.text in result.compressed


@settings(max_examples=40, deadline=None)
@given(
    template=st.sampled_from(CURATED),
    ratio=st.floats(min_value=0.2, max_value=0.9),
)
def test_property_numbers_and_negation_preserved(template, ratio):
    result = compress(template, target_ratio=ratio, min_tokens=0)
    assert _numbers_multiset(template) == _numbers_multiset(result.compressed)
    assert len(extract_negations(result.compressed)) >= len(extract_negations(template))
```

- [ ] **Step 3: Run the invariant suite**

Run: `python -m pytest tests/test_protect_invariants.py -v`
Expected: PASS. **If any invariant fails, it is a real protect-list gap — do NOT weaken the test.** Report the failing (doc, ratio) and the specific lost token; the fix belongs in `protect.py`'s veto/extractors, then re-run. Deletion must not proceed past a failing invariant.

- [ ] **Step 4: Commit**

```bash
git add tests/test_protect_invariants.py data/samples/policy_snip.txt data/samples/notice_snip.txt
git commit -m "test: add the protect-list invariant suite over a corpus"
```

---

## Task 6: Grammaticality, determinism, idempotence

**Files:**
- Create: `tests/test_grammaticality.py`
- Create: `tests/test_determinism.py`

**Interfaces:**
- Consumes: `compress`, `parse`, `segment`.
- Produces: structural-quality guards.

- [ ] **Step 1: Write the tests**

```python
# tests/test_grammaticality.py
import pytest

from promptcomp import compress
from promptcomp.parse import parse
from promptcomp.segment import segment
from promptcomp.types import BlockKind

DOCS = [
    "The committee approved the annual budget in the morning after a long debate. "
    "The vendor, acting in good faith, delivered the goods quickly and without delay.",
    "The Company shall reimburse expenses of $2,500 within thirty days, provided that "
    "receipts are submitted, and no exception applies unless approved in writing.",
]


@pytest.mark.parametrize("doc", DOCS)
@pytest.mark.parametrize("ratio", [0.3, 0.5, 0.7])
def test_every_output_sentence_has_a_root(doc, ratio):
    result = compress(doc, target_ratio=ratio, min_tokens=0)
    for block in segment(result.compressed):
        if block.kind is not BlockKind.PROSE:
            continue
        parsed = parse(block.text)
        for sent in parsed.sents:
            if sent.text.strip():
                assert any(t.dep_ == "ROOT" for t in sent), f"no ROOT in: {sent.text!r}"
```

```python
# tests/test_determinism.py
from promptcomp import compress

DOC = (
    "The committee approved the annual budget in the morning after a long debate, "
    "and the vendor delivered the goods quickly, in good faith, without any delay."
)


def test_compress_is_byte_identical_across_calls(tmp_path):
    r1 = compress(DOC, target_ratio=0.5, min_tokens=0, cache_path=tmp_path / "a.sqlite")
    r2 = compress(DOC, target_ratio=0.5, min_tokens=0, cache_path=tmp_path / "b.sqlite")
    assert r1.compressed == r2.compressed
    assert r1.config_hash == r2.config_hash


def test_idempotence_second_pass_deletes_little(tmp_path):
    r1 = compress(DOC, target_ratio=0.5, min_tokens=0, cache_path=tmp_path / "a.sqlite")
    r2 = compress(r1.compressed, target_ratio=0.5, min_tokens=0, cache_path=tmp_path / "b.sqlite")
    # second pass should not delete a large additional fraction
    assert r2.tokens_after >= 0.7 * r1.tokens_after
```

- [ ] **Step 2: Run the tests**

Run: `python -m pytest tests/test_grammaticality.py tests/test_determinism.py -v`
Expected: PASS. If a grammaticality test fails, a deleted subtree left a rootless remnant — investigate the deprel; the fix is to remove that deprel from `SAFE_DEPRELS` or handle the seam, not to weaken the test.

- [ ] **Step 3: Commit**

```bash
git add tests/test_grammaticality.py tests/test_determinism.py
git commit -m "test: add grammaticality, determinism, and idempotence guards"
```

---

## Task 7: inspect.py shows real deletions

**Files:**
- Modify: `scripts/inspect.py`
- Test: `tests/test_inspect.py`

**Interfaces:**
- Consumes: `compress` (now populates `deleted`/`vetoed`).
- Produces: the normal summary now prints a `deleted :` section (each deletion's deprel + offset + text) drawn from `result.deleted`, so the real cuts are visible alongside the existing `--show-candidates` audit view.

- [ ] **Step 1: Write the failing test (append to tests/test_inspect.py)**

```python
# tests/test_inspect.py  (append)
def test_inspect_prints_deletions(tmp_path, capsys):
    doc = tmp_path / "doc.txt"
    doc.write_text(
        "The committee approved the annual budget in the morning after a long debate.\n"
    )
    rc = inspect_cli.main(
        ["--file", str(doc), "--ratio", "0.5", "--min-tokens", "0",
         "--cache-path", str(tmp_path / "c.sqlite")]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "deleted" in out.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_inspect.py::test_inspect_prints_deletions -v`
Expected: FAIL — no `deleted` section printed.

- [ ] **Step 3: Add the deletions section to inspect.py**

After the substitutions-printing block (and before the `--show-candidates` block) in `main`, add:

```python
    if result.deleted:
        print("deleted       :")
        for d in result.deleted:
            print(f"    [{d.deprel:9}] {d.score:6.3f} {d.span.start:>6}: {d.text!r}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_inspect.py -v`
Expected: PASS.

- [ ] **Step 5: Verify on the real sample document**

Run: `python scripts/inspect.py --file data/samples/memo1.txt --ratio 0.6`
Expected: prints token counts (after < before), a `deleted :` section listing removed adjuncts by deprel, and substitutions. Sanity-check: no deleted span is a dollar amount, date, obligation word, or negation.

- [ ] **Step 6: Run the full suite**

Run: `python -m pytest -v`
Expected: PASS — all tests green, including the invariant suite.

- [ ] **Step 7: Commit**

```bash
git add scripts/inspect.py tests/test_inspect.py
git commit -m "feat: show real deletions in the inspection CLI"
```

---

## Self-Review Notes

**Spec coverage (M6):**
- Stage 3 `Scorer` protocol + `HeuristicScorer` (content density, rarity proxy, deprel prior; deterministic, CPU-only) → Task 1. ✓
- Delete cheapest until budget met; never exceed `max_deletion_fraction` → Tasks 2–3. ✓
- Whole-subtree deletion, grammatical remnant by construction; non-projective guard → Task 2 (`is_contiguous_subtree`), Task 6 (grammaticality). ✓
- End-to-end `compress()` with the stage order structure→veto→rank→substitute → Task 4. ✓
- The invariant suite (numbers multiset, negation non-decreasing, no protected token deleted, passthrough integrity, determinism, idempotence, grammaticality) over a corpus → Tasks 5–6. ✓
- Deletions colored by deprel in inspect → Task 7. ✓
- `deleted`/`vetoed`/`protect_list_version` populated for audit → Task 4. ✓
- `config_hash` includes scorer identity (Plan A/B carryover) → Task 4. ✓
- Invariants reuse shared extractors (B2 carryover) → Task 5. ✓

**Determinism:** selection sorts by `(score, char_start)`; scorer is pure; spaCy deterministic; `test_determinism.py` locks byte-identical output.

**Safety-by-construction:** only veto survivors reach scoring; contiguity guard prevents non-projective slices; the invariant suite is the build gate — a failure means a protect-list gap, fixed in `protect.py`, never by weakening a test.

**Placeholder scan:** no TBD/TODO; every code step contains complete code. The two new corpus files are content. Budget is per-block (a documented, simpler choice than global); global budgeting is a future refinement if the eval (M7) shows ratio drift.
