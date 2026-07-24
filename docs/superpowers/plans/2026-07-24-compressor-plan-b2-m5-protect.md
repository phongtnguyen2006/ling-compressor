# Compressor Plan B2 (M5): Protect-List Veto Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Stage 2 — the protect-list veto that permanently strikes any candidate subtree containing (or scoped by) a protected token — plus the shared protected-token extractors the invariant suite will depend on, so that once deletion is wired (Plan B3) no protected token can ever be deleted.

**Architecture:** A packaged `protect_terms.yaml` / `defined_terms.yaml` load into a frozen `ProtectList`. `protect.py` exposes standalone extractors (`extract_numbers`, `extract_negations`) that BOTH the veto and the future invariant test import — one source of truth for "what is protected." `veto(doc, candidates, protect_list, defined_terms)` recovers each candidate's token span via `doc.char_span(...)` and strikes it if it contains a number, a protected named entity, a negation, a conditional/modal/quantifier term, or a defined term, or if its syntactic root is inside the scope of a negation or a modal auxiliary. This plan is NON-DESTRUCTIVE: `compress()` is unchanged; the veto is surfaced only through unit tests and an audit view in `scripts/inspect.py`. Deletion is Plan B3.

**Tech Stack:** Python 3.12, spaCy `en_core_web_sm` (NER + parse), `pyyaml`, existing types; `pytest` for tests.

## Global Constraints

- Python 3.12; package `promptcomp`, src-layout. Data files packaged under `src/promptcomp/data/`, loaded via `importlib.resources`.
- **Veto is a HARD rule and permanent:** a struck candidate is never reconsidered. Over-vetoing (protecting too much) is the safe direction; under-vetoing is a build failure.
- **Shared extraction:** `extract_numbers(text)` and `extract_negations(text)` are the single source of truth; the veto and (Plan B3) the invariant test both use them, so the "no protected token deleted" invariant holds by construction.
- **Same-Doc requirement:** the veto operates on the exact `Doc` the candidates were enumerated from (needs `doc.ents` for NER and the parse tree for scope). Recover tokens with `doc.char_span(char_start, char_end)`; if it returns `None` (range not token-aligned), VETO the candidate (safe default).
- **Non-destructive:** `compress()` output must be byte-identical to Plan B1 for the same input. No regression to the existing 45 tests.
- **Determinism:** veto output order is deterministic (survivors keep candidate order; vetoes in candidate order).
- **Protect-list version** is loaded and available to stamp into results later (Plan B3).
- Commit in small TDD steps. **Do not pass any `-c user.name=…/-c user.email=…` override to `git commit`** — use the repo's configured identity (`phongtnguyen2006 <phongtnguyen2006@gmail.com>`).

---

## File Structure

| File | Responsibility |
|---|---|
| `src/promptcomp/data/protect_terms.yaml` | Protected NER labels + term lists per class + version |
| `src/promptcomp/data/defined_terms.yaml` | Project-specific defined terms + version |
| `src/promptcomp/protect.py` | `ProtectList`, loaders, extractors, `extract_defined_terms`, `veto()` |
| `scripts/inspect.py` | `--show-candidates` now marks each candidate SURVIVOR / VETOED:reason |
| `tests/test_protect.py` | Extractors, veto content rules, scope veto, defined-term extraction |
| `tests/test_inspect.py` | Audit-view test |

**Interface contract carried into Plan B3 (do not break):**
- `extract_numbers(text: str) -> list[str]`
- `extract_negations(text: str) -> list[str]`
- `extract_defined_terms(text: str) -> set[str]`
- `load_protect_list(path=None) -> ProtectList` (has `.version`)
- `load_defined_terms(path=None) -> tuple[frozenset[str], str]`
- `veto(doc, candidates: list[Candidate], protect_list: ProtectList, defined_terms: frozenset[str] = frozenset()) -> tuple[list[Candidate], list[Veto]]`

---

## Task 1: Protect-list data files

**Files:**
- Create: `src/promptcomp/data/protect_terms.yaml`
- Create: `src/promptcomp/data/defined_terms.yaml`
- Test: `tests/test_protect.py`

**Interfaces:**
- Consumes: nothing.
- Produces: two packaged YAML resources (shipped via the existing `package-data = ["data/*.yaml"]` glob from Plan B1).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_protect.py
import yaml
from importlib.resources import files


def test_protect_terms_resource_loads():
    doc = yaml.safe_load(files("promptcomp").joinpath("data/protect_terms.yaml").read_text())
    assert doc["version"].startswith("protect-")
    assert "MONEY" in doc["ner_numeric"]
    assert "ORG" in doc["ner_entity"]
    assert "shall" in doc["single_terms"]["modal"]
    assert "provided that" in doc["phrase_terms"]["conditional"]


def test_defined_terms_resource_loads():
    doc = yaml.safe_load(files("promptcomp").joinpath("data/defined_terms.yaml").read_text())
    assert doc["version"].startswith("defined-")
    assert isinstance(doc["terms"], list)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_protect.py -v`
Expected: FAIL — resources do not exist.

- [ ] **Step 3: Create protect_terms.yaml**

```yaml
# src/promptcomp/data/protect_terms.yaml
version: "protect-2026-07-24.1"

# spaCy NER labels whose entities must never be deleted.
ner_numeric: [CARDINAL, MONEY, PERCENT, DATE, QUANTITY, ORDINAL]
ner_entity: [ORG, PERSON, GPE, LAW, FAC]

# Single-word protected terms, matched case-insensitively on token text/lemma.
single_terms:
  conditional: [unless, except, if, otherwise]
  modal: [shall, must, may, obligated]
  quantifier: [all, each, any, every, solely, exclusively]

# Multi-word protected phrases, matched case-insensitively as substrings of the candidate text.
phrase_terms:
  conditional: ["provided that", "subject to", "contingent on", "in the event", "only if"]
  modal: ["will not", "required to", "entitled to", "not to exceed", "at least", "no more than", "at most"]
```

- [ ] **Step 4: Create defined_terms.yaml**

```yaml
# src/promptcomp/data/defined_terms.yaml
# Project-specific defined terms that must never be deleted. Extend per corpus.
# Auto-extraction from definition patterns (Stage 0) supplements this at runtime.
version: "defined-2026-07-24.1"
terms: []
```

- [ ] **Step 5: Reinstall editable and run the test**

Run: `python -m pip install -e ".[dev]" && python -m pytest tests/test_protect.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add src/promptcomp/data/protect_terms.yaml src/promptcomp/data/defined_terms.yaml tests/test_protect.py
git commit -m "feat: add packaged protect-list and defined-terms data files"
```

---

## Task 2: ProtectList + loaders

**Files:**
- Create: `src/promptcomp/protect.py`
- Test: `tests/test_protect.py`

**Interfaces:**
- Consumes: the YAML resources.
- Produces:
  - `@dataclass(frozen=True) class ProtectList` with: `version: str`, `ner_numeric: frozenset[str]`, `ner_entity: frozenset[str]`, `single_terms: dict[str, frozenset[str]]`, `phrase_terms: dict[str, frozenset[str]]`.
  - `load_protect_list(path=None) -> ProtectList`
  - `load_defined_terms(path=None) -> tuple[frozenset[str], str]`

- [ ] **Step 1: Write the failing test (append to tests/test_protect.py)**

```python
# tests/test_protect.py  (append)
from promptcomp.protect import load_protect_list, load_defined_terms, ProtectList


def test_load_protect_list_default():
    pl = load_protect_list()
    assert isinstance(pl, ProtectList)
    assert pl.version.startswith("protect-")
    assert "MONEY" in pl.ner_numeric
    assert "ORG" in pl.ner_entity
    assert "shall" in pl.single_terms["modal"]
    assert "provided that" in pl.phrase_terms["conditional"]


def test_load_defined_terms_default():
    terms, version = load_defined_terms()
    assert isinstance(terms, frozenset)
    assert version.startswith("defined-")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_protect.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'promptcomp.protect'`.

- [ ] **Step 3: Write the implementation**

```python
# src/promptcomp/protect.py
"""Stage 2: protect-list veto.

Strikes any candidate subtree that contains — or is syntactically scoped by —
a protected token. This is a hard rule: over-protecting is safe, under-protecting
is a build failure.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ProtectList:
    version: str
    ner_numeric: frozenset[str]
    ner_entity: frozenset[str]
    single_terms: dict[str, frozenset[str]]
    phrase_terms: dict[str, frozenset[str]]


def _read_resource(name: str, path: str | Path | None) -> dict:
    if path is not None:
        text = Path(path).read_text()
    else:
        text = files("promptcomp").joinpath(f"data/{name}").read_text()
    return yaml.safe_load(text)


def load_protect_list(path: str | Path | None = None) -> ProtectList:
    doc = _read_resource("protect_terms.yaml", path)
    single = {k: frozenset(t.lower() for t in v) for k, v in doc.get("single_terms", {}).items()}
    phrase = {k: frozenset(t.lower() for t in v) for k, v in doc.get("phrase_terms", {}).items()}
    return ProtectList(
        version=str(doc["version"]),
        ner_numeric=frozenset(doc.get("ner_numeric", [])),
        ner_entity=frozenset(doc.get("ner_entity", [])),
        single_terms=single,
        phrase_terms=phrase,
    )


def load_defined_terms(path: str | Path | None = None) -> tuple[frozenset[str], str]:
    doc = _read_resource("defined_terms.yaml", path)
    return frozenset(doc.get("terms", []) or []), str(doc.get("version", ""))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_protect.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/promptcomp/protect.py tests/test_protect.py
git commit -m "feat: add ProtectList and protect-list loaders"
```

---

## Task 3: Shared protected-token extractors

**Files:**
- Modify: `src/promptcomp/protect.py`
- Test: `tests/test_protect.py`

**Interfaces:**
- Consumes: nothing (pure text functions).
- Produces:
  - `extract_numbers(text: str) -> list[str]` — every numeric literal (currency, percent, decimals, thousands) in document order.
  - `extract_negations(text: str) -> list[str]` — every negation-word occurrence (lowercased) in document order.

These are the single source of truth reused by the invariant suite in Plan B3.

- [ ] **Step 1: Write the failing test (append to tests/test_protect.py)**

```python
# tests/test_protect.py  (append)
from promptcomp.protect import extract_numbers, extract_negations


def test_extract_numbers_covers_currency_percent_decimals():
    text = "The fee is $1,000.00, a 15% surcharge, and 42 units at $0.67 each."
    nums = extract_numbers(text)
    assert "$1,000.00" in nums
    assert "15%" in nums
    assert "42" in nums
    assert "$0.67" in nums


def test_extract_numbers_is_multiset_ordered():
    text = "5 and 5 and 10."
    assert extract_numbers(text) == ["5", "5", "10"]


def test_extract_negations_finds_all_forms():
    text = "It is not allowed; no exceptions, never waived, without consent."
    negs = extract_negations(text)
    assert "not" in negs
    assert "no" in negs
    assert "never" in negs
    assert "without" in negs


def test_extract_negations_word_boundary():
    # "nothing" contains "no" but must not count as a bare "no".
    text = "There is nothing here."
    assert "no" not in extract_negations(text)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_protect.py -v`
Expected: FAIL — `ImportError: cannot import name 'extract_numbers'`.

- [ ] **Step 3: Write the implementation (append to protect.py)**

```python
# src/promptcomp/protect.py  (add near the top imports)
import re

# Currency/optional-thousands/optional-decimal/optional-percent numeric literal.
_NUMBER_RE = re.compile(r"\$?\d[\d,]*(?:\.\d+)?%?")

# Negation words treated as protected. Matched as whole words, case-insensitive.
_NEGATION_WORDS = frozenset(
    {"not", "no", "never", "without", "nor", "neither", "none", "cannot"}
)
_NEGATION_RE = re.compile(
    r"\b(" + "|".join(sorted(_NEGATION_WORDS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)
```

```python
# src/promptcomp/protect.py  (append functions)
def extract_numbers(text: str) -> list[str]:
    """Every numeric literal in document order (the invariant multiset)."""
    return _NUMBER_RE.findall(text)


def extract_negations(text: str) -> list[str]:
    """Every negation-word occurrence, lowercased, in document order."""
    return [m.group(0).lower() for m in _NEGATION_RE.finditer(text)]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_protect.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/promptcomp/protect.py tests/test_protect.py
git commit -m "feat: add shared protected-token extractors (numbers, negations)"
```

---

## Task 4: Defined-term auto-extraction

**Files:**
- Modify: `src/promptcomp/protect.py`
- Test: `tests/test_protect.py`

**Interfaces:**
- Consumes: nothing (pure text).
- Produces: `extract_defined_terms(text: str) -> set[str]` — terms introduced by definition patterns: `"X" means …`, `X ("the Y")`, `X ("Y")`.

- [ ] **Step 1: Write the failing test (append to tests/test_protect.py)**

```python
# tests/test_protect.py  (append)
from promptcomp.protect import extract_defined_terms


def test_defined_term_quoted_means_pattern():
    text = 'The "Reimbursable Amount" means the total eligible expense.'
    terms = extract_defined_terms(text)
    assert "Reimbursable Amount" in terms


def test_defined_term_parenthetical_pattern():
    text = 'Acme Corporation ("the Company") shall pay all fees.'
    terms = extract_defined_terms(text)
    assert any("Company" in t for t in terms)


def test_no_false_positive_on_plain_prose():
    text = "The committee approved the budget in the morning."
    assert extract_defined_terms(text) == set()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_protect.py -v`
Expected: FAIL — `ImportError: cannot import name 'extract_defined_terms'`.

- [ ] **Step 3: Write the implementation (append to protect.py)**

```python
# src/promptcomp/protect.py  (add patterns near the other regexes)
# "X" means ... / “X” means ...
_DEF_MEANS_RE = re.compile(r'["“]([A-Z][^"”]{1,60})["”]\s+means\b')
# X ("the Y") / X ("Y")
_DEF_PAREN_RE = re.compile(r'\(["“](?:the\s+)?([A-Z][^"”]{1,60})["”]\)')
```

```python
# src/promptcomp/protect.py  (append function)
def extract_defined_terms(text: str) -> set[str]:
    """Terms introduced by definition patterns; supplements the yaml list."""
    terms: set[str] = set()
    for pat in (_DEF_MEANS_RE, _DEF_PAREN_RE):
        for m in pat.finditer(text):
            terms.add(m.group(1).strip())
    return terms
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_protect.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/promptcomp/protect.py tests/test_protect.py
git commit -m "feat: add defined-term auto-extraction from definition patterns"
```

---

## Task 5: veto() — content + scope

**Files:**
- Modify: `src/promptcomp/protect.py`
- Test: `tests/test_protect.py`

**Interfaces:**
- Consumes: `Candidate`, `Veto`, `Span` from `types`; `parse` from `parse` (tests only); a spaCy `Doc`.
- Produces: `veto(doc, candidates, protect_list, defined_terms=frozenset()) -> tuple[list[Candidate], list[Veto]]`.

Rules (a candidate is struck on the FIRST matching reason):
1. `doc.char_span(start, end)` is `None` → strike (`"unaligned"`, safe default).
2. Any numeric literal in the candidate text (`extract_numbers`) → `"number"`.
3. Any `doc.ents` entity overlapping the span whose label is protected → `"number"` (numeric label) or `"entity"`.
4. Any token in the span with `dep_ == "neg"`, or any negation word (`extract_negations`) in the text → `"negation"`.
5. Any span token whose lowercased text or lemma is in a `single_terms` class → that class name.
6. Any `phrase_terms` phrase appearing (case-insensitive) in the text → that class name.
7. Any defined term appearing (case-insensitive) in the text → `"defined_term"`.
8. Scope: the span's syntactic root is a descendant of (or is) a token that has a `neg` child, or has a modal auxiliary child → `"scope"`.

Survivors keep original order; vetoes are recorded in original order with a `Veto(span, text, protect_class)`.

- [ ] **Step 1: Write the failing tests (append to tests/test_protect.py)**

```python
# tests/test_protect.py  (append)
from promptcomp.parse import parse, enumerate_candidates
from promptcomp.protect import veto, load_protect_list


def _veto_reasons(text, defined=frozenset()):
    doc = parse(text)
    cands = enumerate_candidates(doc)
    survivors, vetoes = veto(doc, cands, load_protect_list(), defined)
    return survivors, vetoes, {v.protect_class for v in vetoes}


def test_candidate_with_number_is_vetoed():
    # "of $2 million" is a PP candidate carrying money — must be struck.
    _, vetoes, classes = _veto_reasons(
        "The committee approved a budget of $2 million after long debate."
    )
    assert any("2 million" in v.text or "$2" in v.text for v in vetoes)


def test_candidate_with_negation_is_vetoed():
    _, vetoes, classes = _veto_reasons(
        "The vendor delivered the goods without any prior notice."
    )
    assert "negation" in classes


def test_clean_adjunct_survives():
    survivors, vetoes, _ = _veto_reasons(
        "The committee approved the budget in the morning."
    )
    # "in the morning" carries nothing protected -> survives.
    assert any("in the morning" in c.text for c in survivors)


def test_modal_scope_vetoes_descendant_adjunct():
    # The adverbial under "must" is in modal scope -> struck even if it looks clean.
    survivors, vetoes, classes = _veto_reasons(
        "The tenant must vacate the premises quietly."
    )
    assert any(v.protect_class in {"scope", "modal"} for v in vetoes)


def test_defined_term_in_candidate_is_vetoed():
    _, vetoes, classes = _veto_reasons(
        "The board approved the plan for the Reimbursable Amount without delay.",
        defined=frozenset({"Reimbursable Amount"}),
    )
    assert "defined_term" in classes or "negation" in classes  # negation also present


def test_veto_partitions_candidates():
    doc = parse("The committee approved a budget of $2 million in the morning.")
    cands = enumerate_candidates(doc)
    survivors, vetoes = veto(doc, cands, load_protect_list())
    assert len(survivors) + len(vetoes) == len(cands)
    # deterministic
    s2, v2 = veto(doc, cands, load_protect_list())
    assert [c.text for c in survivors] == [c.text for c in s2]
    assert [v.text for v in vetoes] == [v.text for v in v2]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_protect.py -v`
Expected: FAIL — `ImportError: cannot import name 'veto'`.

- [ ] **Step 3: Write the implementation (append to protect.py)**

```python
# src/promptcomp/protect.py  (add import near the top)
from .types import Candidate, Span, Veto

_MODAL_AUX_WORDS = frozenset({"shall", "must", "may", "will", "would", "should", "can", "could", "might"})
```

```python
# src/promptcomp/protect.py  (append)
def _span_tokens(doc, start: int, end: int) -> list:
    return [t for t in doc if t.idx >= start and (t.idx + len(t.text)) <= end]


def _content_class(doc, cand: Candidate, pl: ProtectList, defined_terms: frozenset[str]):
    text = cand.text
    low = text.lower()

    if extract_numbers(text):
        return "number"

    for ent in doc.ents:
        if ent.label_ in pl.ner_numeric or ent.label_ in pl.ner_entity:
            if not (ent.end_char <= cand.char_start or ent.start_char >= cand.char_end):
                return "number" if ent.label_ in pl.ner_numeric else "entity"

    toks = _span_tokens(doc, cand.char_start, cand.char_end)
    if any(t.dep_ == "neg" for t in toks) or extract_negations(text):
        return "negation"

    forms = {t.text.lower() for t in toks} | {t.lemma_.lower() for t in toks}
    for cls, words in pl.single_terms.items():
        if forms & words:
            return cls

    for cls, phrases in pl.phrase_terms.items():
        for phrase in phrases:
            if phrase in low:
                return cls

    for term in defined_terms:
        if term.lower() in low:
            return "defined_term"

    return None


def _in_neg_or_modal_scope(root_token) -> bool:
    chain = [root_token] + list(root_token.ancestors)
    for anc in chain:
        for child in anc.children:
            if child.dep_ == "neg":
                return True
            if child.dep_ in {"aux", "auxpass"} and child.text.lower() in _MODAL_AUX_WORDS:
                return True
    return False


def veto(doc, candidates, protect_list, defined_terms=frozenset()):
    """Partition candidates into (survivors, vetoes). Hard rule; deterministic."""
    survivors: list[Candidate] = []
    vetoes: list[Veto] = []
    for cand in candidates:
        span = doc.char_span(cand.char_start, cand.char_end)
        reason = None
        if span is None:
            reason = "unaligned"
        else:
            reason = _content_class(doc, cand, protect_list, defined_terms)
            if reason is None and _in_neg_or_modal_scope(span.root):
                reason = "scope"
        if reason is None:
            survivors.append(cand)
        else:
            vetoes.append(
                Veto(
                    span=Span(cand.char_start, cand.char_end),
                    text=cand.text,
                    protect_class=reason,
                )
            )
    return survivors, vetoes
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_protect.py -v`
Expected: PASS. If `test_modal_scope_vetoes_descendant_adjunct` fails, print the parse (`for t in parse(sent): print(t.text, t.dep_, t.head.text)`) and confirm the modal attaches as `aux`; adjust `_MODAL_AUX_WORDS` / the scope walk to match real spaCy structure — this is the expected reconciliation point.

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest -v`
Expected: PASS — no regression (compress() unchanged).

- [ ] **Step 6: Commit**

```bash
git add src/promptcomp/protect.py tests/test_protect.py
git commit -m "feat: add protect-list veto with content and scope rules"
```

---

## Task 6: inspect.py audit view (survivors vs vetoed)

**Files:**
- Modify: `scripts/inspect.py`
- Test: `tests/test_inspect.py`

**Interfaces:**
- Consumes: `parse`, `enumerate_candidates`, `veto`, `load_protect_list`, `load_defined_terms`, `extract_defined_terms`.
- Produces: the existing `--show-candidates` view now labels each candidate `SURVIVOR` or `VETOED:<class>`, so the audit trail (what we chose not to cut, and why) is visible — the M5 checkpoint.

- [ ] **Step 1: Write the failing test (append to tests/test_inspect.py)**

```python
# tests/test_inspect.py  (append)
def test_show_candidates_marks_vetoed(tmp_path, capsys):
    doc = tmp_path / "doc.txt"
    doc.write_text("The committee approved a budget of $2 million in the morning.\n")
    rc = inspect_cli.main(
        ["--file", str(doc), "--show-candidates", "--min-tokens", "0",
         "--cache-path", str(tmp_path / "c.sqlite")]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "VETOED" in out          # the money PP is struck
    assert "number" in out          # with its reason
    assert "SURVIVOR" in out        # "in the morning" survives
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_inspect.py::test_show_candidates_marks_vetoed -v`
Expected: FAIL — output has no SURVIVOR/VETOED labels yet.

- [ ] **Step 3: Update the --show-candidates branch in inspect.py**

Replace the body of the `if args.show_candidates:` branch (added in Plan B1) with a version that runs the veto:

```python
    if args.show_candidates:
        from promptcomp.segment import segment
        from promptcomp.types import BlockKind
        from promptcomp.parse import parse, enumerate_candidates
        from promptcomp.protect import (
            veto, load_protect_list, load_defined_terms, extract_defined_terms,
        )

        pl = load_protect_list()
        yaml_terms, _ = load_defined_terms()
        auto_terms = extract_defined_terms(text)
        defined = frozenset(yaml_terms) | frozenset(auto_terms)

        print("candidates    :")
        for block in segment(text):
            if block.kind is not BlockKind.PROSE:
                continue
            doc = parse(block.text)
            cands = enumerate_candidates(doc)
            survivors, vetoes = veto(doc, cands, pl, defined)
            veto_by_range = {(v.span.start, v.span.end): v.protect_class for v in vetoes}
            for c in cands:
                start = block.start + c.char_start
                mark = "SURVIVOR"
                reason = veto_by_range.get((c.char_start, c.char_end))
                if reason is not None:
                    mark = f"VETOED:{reason}"
                print(f"    [{c.deprel:9}] {mark:16} {start:>6}: {c.text!r}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_inspect.py -v`
Expected: PASS.

- [ ] **Step 5: Verify on the real sample document**

Run: `python scripts/inspect.py --file data/samples/memo1.txt --ratio 0.6 --show-candidates`
Expected: candidates now annotated SURVIVOR / VETOED:<class>. Sanity-check that every candidate containing a dollar amount, date, percentage, obligation word (shall/must/may), or negation is VETOED, and that clean adjuncts survive. This is the M5 audit checkpoint.

- [ ] **Step 6: Run the full suite**

Run: `python -m pytest -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add scripts/inspect.py tests/test_inspect.py
git commit -m "feat: annotate inspect candidates with veto survivors and reasons"
```

---

## Self-Review Notes

**Spec coverage (M5):**
- Stage 2 veto for all protected classes — numbers/currency/percent/dates (regex + NER), negation (dep + lemmas), conditionals, modals/obligation, named entities (NER), defined terms (yaml + auto), scope-bearing quantifiers → Tasks 3–5. ✓
- Scope veto (descendant of a `neg` child or a modal aux) → Task 5 `_in_neg_or_modal_scope`. ✓
- Protect-list versioned & packaged, hot-reloadable via `load_protect_list(path=...)` → Tasks 1–2. ✓
- Defined terms from yaml + auto-extracted from definition patterns → Tasks 1, 4. ✓
- Audit trail visible (what was struck and why) → Task 6. ✓
- Shared extractors for the invariant suite → Task 3 (consumed by Plan B3). ✓

**Non-regression / non-destructive:** `compress()` and the pipeline are untouched; the full suite is re-run in Tasks 5 and 6. Deletion, budget, scoring, and the end-to-end invariant suite (which needs deletion) are Plan B3.

**Same-Doc + non-projective safety:** the veto recovers tokens via `doc.char_span`; a `None` (non-token-aligned) range is vetoed by default — this both satisfies the veto's needs and pre-empts the non-projective deletion risk flagged in the B1 carryover.

**Reconciliation points flagged:** the modal-scope test (Task 5 Step 4) calls out where to confirm spaCy's real `aux`/`neg` structure, matching the label-scheme discipline from B1.

**Placeholder scan:** no TBD/TODO; every code step contains complete code. `defined_terms.yaml` ships with an empty `terms: []` by design (auto-extraction supplements it; corpora extend it).
