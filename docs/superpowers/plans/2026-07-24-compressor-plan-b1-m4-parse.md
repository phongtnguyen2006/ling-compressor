# Compressor Plan B1 (M4): Parse + Candidate Enumeration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Stage 1 (dependency parsing + deletable-subtree enumeration) as a standalone, non-destructive capability — so candidates can be inspected visually before any deletion — and fix the deferred data-packaging issue from Plan A.

**Architecture:** A new `parse.py` lazily loads spaCy `en_core_web_sm` and walks each sentence's dependency tree, emitting a `Candidate` (a frozen dataclass) for every subtree whose root bears an adjunct/modifier dependency label, while never emitting one for a core argument. Enumeration is pure and deterministic. `compress()` is NOT changed in this plan — deletion arrives in Plan B3 (M6); here, `scripts/inspect.py` gains a `--show-candidates` view so the argument/adjunct split can be eyeballed on real text. Separately, the package data file(s) move under `src/promptcomp/data/` and load via `importlib.resources` so the package works when installed as a wheel, not only editable.

**Tech Stack:** Python 3.12, spaCy (`en_core_web_sm`), existing `pyyaml`/`tiktoken`; `pytest`/`hypothesis` for tests.

## Global Constraints

- Python floor 3.12; package `promptcomp`, src-layout at `src/promptcomp/`.
- **spaCy label scheme:** `en_core_web_sm` emits ClearNLP/OntoNotes labels via `token.dep_`, NOT Universal Dependencies. The spec's UD deprels map to spaCy labels as: UD `obl`/`nmod`(PP) → spaCy **`prep`** (with `pobj` as its object); UD `acl:relcl` → spaCy **`relcl`**; UD `obj` → spaCy **`dobj`**; UD `advmod`/`advcl`/`acl`/`appos`/`amod`/`nmod`/`parataxis` keep their names; spaCy root label is **`ROOT`** (uppercase). Verify labels against real `token.dep_` output during implementation — do not assume UD.
- **Enumeration is non-destructive:** this plan never deletes text. `compress()` output must be byte-for-byte identical to Plan A's for the same input (segment + substitute only). No regression to the existing 33 tests.
- **Determinism:** candidate lists are sorted deterministically (`(char_start, char_end)`); the spaCy pipeline is deterministic; no sampling.
- **Whole subtrees only:** a candidate's char span is the full span of its root token's `subtree`. Never a partial span.
- **Core arguments are never candidates:** `nsubj`, `nsubjpass`, `csubj`, `dobj`, `dative`, `iobj`, `ccomp`, `xcomp`, `ROOT`, `aux`, `auxpass`, `cop`, `pobj` must never appear as a candidate's `deprel`.
- **Packaging:** package data loads via `importlib.resources`, not a repo-relative path, so `compress()` works under a non-editable install.
- Commit in small TDD steps. **Do not pass any `-c user.name=…/-c user.email=…` override to `git commit`** — use the repository's configured identity (already set to `phongtnguyen2006 <phongtnguyen2006@gmail.com>`).

---

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | Add `[tool.setuptools.package-data]` so `data/*.yaml` ships with the package |
| `src/promptcomp/data/substitutions.yaml` | Moved here from repo-root `data/` (packaged resource) |
| `src/promptcomp/substitute.py` | Load substitutions via `importlib.resources` instead of a repo-relative path |
| `src/promptcomp/types.py` | Add the `Candidate` frozen dataclass |
| `src/promptcomp/parse.py` | Stage 1: `load_nlp()`, `parse(text)`, `enumerate_candidates(doc)` |
| `scripts/inspect.py` | Add `--show-candidates` view |
| `tests/test_substitute.py` | Repoint the packaged-resource test |
| `tests/test_parse.py` | Parse + enumeration behavior tests |
| `tests/test_inspect.py` | `--show-candidates` test |

**Interface contract carried into Plan B2/B3 (do not break):**
- `Candidate(deprel: str, char_start: int, char_end: int, head_text: str, text: str, n_tokens: int)` — char offsets are relative to the parsed text passed to `parse()`.
- `parse(text: str) -> spacy.tokens.Doc`
- `enumerate_candidates(doc) -> list[Candidate]`
- `load_nlp() -> spacy.language.Language` (cached singleton)

---

## Task 1: Package the data file(s) via importlib.resources

**Files:**
- Modify: `pyproject.toml`
- Move: `data/substitutions.yaml` → `src/promptcomp/data/substitutions.yaml`
- Modify: `src/promptcomp/substitute.py`
- Modify: `tests/test_substitute.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `load_substitutions(path=None)` unchanged in signature, but the default now reads the packaged resource `promptcomp/data/substitutions.yaml`.

- [ ] **Step 1: Write the failing test (replace the old path-based test)**

Open `tests/test_substitute.py`. Replace the existing `test_substitutions_file_loads_and_has_phrases` (which reads `Path(__file__).parents[1] / "data" / "substitutions.yaml"`) with a packaged-resource test:

```python
# tests/test_substitute.py  (replace the old file-path test with this)
from importlib.resources import files


def test_substitutions_file_is_packaged():
    data = files("promptcomp").joinpath("data/substitutions.yaml").read_text()
    doc = yaml.safe_load(data)
    assert doc["version"].startswith("sub-")
    assert doc["phrases"]["in order to"] == "to"
    assert doc["phrases"]["due to the fact that"] == "because"
```

Keep the existing `import yaml` at the top of the file (add it if the removal left it unused elsewhere).

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_substitute.py::test_substitutions_file_is_packaged -v`
Expected: FAIL — the resource `promptcomp/data/substitutions.yaml` does not exist yet.

- [ ] **Step 3: Move the data file into the package**

Run:
```bash
mkdir -p src/promptcomp/data
git mv data/substitutions.yaml src/promptcomp/data/substitutions.yaml
```

- [ ] **Step 4: Update substitute.py to load via importlib.resources**

In `src/promptcomp/substitute.py`, remove the `_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "substitutions.yaml"` line and change `load_substitutions` to read the packaged resource:

```python
# src/promptcomp/substitute.py  (imports)
from importlib.resources import files
# (Path may still be imported for the explicit-path branch below)
```

```python
# src/promptcomp/substitute.py  (replace load_substitutions)
def load_substitutions(path: str | Path | None = None) -> tuple[dict[str, str], str]:
    if path is not None:
        text = Path(path).read_text()
    else:
        text = files("promptcomp").joinpath("data/substitutions.yaml").read_text()
    doc = yaml.safe_load(text)
    return dict(doc.get("phrases", {})), str(doc.get("version", ""))
```

- [ ] **Step 5: Add package-data to pyproject.toml**

In `pyproject.toml`, add this section (below the existing `[tool.setuptools.packages.find]`):

```toml
[tool.setuptools.package-data]
promptcomp = ["data/*.yaml"]
```

- [ ] **Step 6: Reinstall editable and run the full suite**

Run: `python -m pip install -e ".[dev]" && python -m pytest -v`
Expected: PASS — all tests (the repointed packaged-resource test plus the existing suite; `test_load_substitutions_default_path` still passes because `load_substitutions()` now reads the packaged copy). Count should remain 33.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/promptcomp/data/substitutions.yaml src/promptcomp/substitute.py tests/test_substitute.py
git commit -m "fix: package data files and load substitutions via importlib.resources"
```

---

## Task 2: Add the Candidate type

**Files:**
- Modify: `src/promptcomp/types.py`
- Test: `tests/test_types.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `@dataclass(frozen=True) class Candidate(deprel: str, char_start: int, char_end: int, head_text: str, text: str, n_tokens: int)`.

- [ ] **Step 1: Write the failing test (append to tests/test_types.py)**

```python
# tests/test_types.py  (append)
def test_candidate_fields_and_frozen():
    from promptcomp.types import Candidate
    c = Candidate(
        deprel="prep", char_start=10, char_end=25,
        head_text="in", text="in the morning", n_tokens=3,
    )
    assert c.deprel == "prep"
    assert c.char_start == 10 and c.char_end == 25
    assert c.head_text == "in" and c.text == "in the morning"
    assert c.n_tokens == 3
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.deprel = "advmod"
```

(`pytest` and `dataclasses` are already imported at the top of `tests/test_types.py` from Plan A.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_types.py::test_candidate_fields_and_frozen -v`
Expected: FAIL — `ImportError: cannot import name 'Candidate'`.

- [ ] **Step 3: Add the dataclass to types.py**

Append to `src/promptcomp/types.py` (after the existing `Substitution` dataclass):

```python
@dataclass(frozen=True)
class Candidate:
    deprel: str
    char_start: int
    char_end: int
    head_text: str
    text: str
    n_tokens: int
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_types.py -v`
Expected: PASS (existing type tests + the new one).

- [ ] **Step 5: Commit**

```bash
git add src/promptcomp/types.py tests/test_types.py
git commit -m "feat: add Candidate type for deletable subtrees"
```

---

## Task 3: spaCy loader + parse()

**Files:**
- Create: `src/promptcomp/parse.py`
- Test: `tests/test_parse.py`

**Interfaces:**
- Consumes: nothing (loads spaCy).
- Produces:
  - `load_nlp() -> spacy.language.Language` — cached module singleton; loads `en_core_web_sm`; raises an actionable `RuntimeError` if spaCy or the model is missing.
  - `parse(text: str) -> spacy.tokens.Doc`.

- [ ] **Step 1: Install the spaCy English model**

Run: `python -m spacy download en_core_web_sm`
Expected: downloads and installs `en_core_web_sm` (a one-time setup; the tests below require it).

- [ ] **Step 2: Confirm the label scheme (diagnostic, not committed)**

Run:
```bash
python -c "import spacy; nlp=spacy.load('en_core_web_sm'); [print(t.text, t.dep_, t.head.text) for t in nlp('The committee approved the budget in the morning.')]"
```
Expected: you will see spaCy labels like `committee nsubj`, `approved ROOT`, `budget dobj`, `in prep`, `morning pobj`. Use these real labels — confirm `prep`/`relcl`/`dobj`/`ROOT`, not UD names — before finalizing the deprel sets in Task 4.

- [ ] **Step 3: Write the failing test**

```python
# tests/test_parse.py
from promptcomp.parse import parse, load_nlp


def test_parse_splits_sentences_and_has_root():
    doc = parse("The cat sat. The dog ran fast.")
    sents = list(doc.sents)
    assert len(sents) == 2
    assert any(t.dep_ == "ROOT" for t in doc)


def test_load_nlp_is_cached_singleton():
    assert load_nlp() is load_nlp()


def test_parse_preserves_text():
    text = "She quickly finished the difficult task."
    doc = parse(text)
    assert doc.text == text
```

- [ ] **Step 4: Run test to verify it fails**

Run: `python -m pytest tests/test_parse.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'promptcomp.parse'`.

- [ ] **Step 5: Write the implementation**

```python
# src/promptcomp/parse.py
"""Stage 1: dependency parsing and deletable-subtree enumeration.

en_core_web_sm emits ClearNLP/OntoNotes dependency labels via token.dep_
(e.g. `prep`, `relcl`, `dobj`, `ROOT`) — NOT Universal Dependencies.
"""
from __future__ import annotations

_NLP = None


def load_nlp():
    """Load and cache the spaCy English pipeline (singleton)."""
    global _NLP
    if _NLP is None:
        try:
            import spacy
        except ImportError as e:
            raise RuntimeError(
                "spaCy is required. Install with: pip install -e '.[dev]' "
                "(spacy is a core dependency)."
            ) from e
        try:
            _NLP = spacy.load("en_core_web_sm")
        except OSError as e:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' is not installed. "
                "Install it with: python -m spacy download en_core_web_sm"
            ) from e
    return _NLP


def parse(text: str):
    """Parse text into a spaCy Doc."""
    return load_nlp()(text)
```

- [ ] **Step 6: Run test to verify it passes**

Run: `python -m pytest tests/test_parse.py -v`
Expected: PASS (3 passed).

- [ ] **Step 7: Commit**

```bash
git add src/promptcomp/parse.py tests/test_parse.py
git commit -m "feat: add spaCy loader and parse() for Stage 1"
```

---

## Task 4: enumerate_candidates()

**Files:**
- Modify: `src/promptcomp/parse.py`
- Test: `tests/test_parse.py`

**Interfaces:**
- Consumes: `Candidate` from `types`; a spaCy `Doc` from `parse`.
- Produces: `enumerate_candidates(doc) -> list[Candidate]`, plus module-level `SAFE_DEPRELS` and `CORE_DEPRELS` frozensets.

Rule: a token whose `dep_` is in `SAFE_DEPRELS` roots a candidate; the candidate span is the full char span of `token.subtree`. Tokens with core-argument labels never root a candidate. Results are sorted by `(char_start, char_end)` for determinism.

- [ ] **Step 1: Write the failing tests (append to tests/test_parse.py)**

```python
# tests/test_parse.py  (append)
from promptcomp.parse import enumerate_candidates, SAFE_DEPRELS, CORE_DEPRELS
from promptcomp.types import Candidate


def test_prep_phrase_is_candidate():
    doc = parse("The committee approved the budget in the morning.")
    cands = enumerate_candidates(doc)
    assert any("in the morning" in c.text for c in cands)


def test_core_arguments_are_never_candidates():
    doc = parse("The committee approved the budget in the morning.")
    cands = enumerate_candidates(doc)
    for c in cands:
        assert c.deprel not in CORE_DEPRELS
    # subject head "committee" and object head "budget" must not head a candidate
    heads = {c.head_text.lower() for c in cands}
    assert "committee" not in heads
    assert "budget" not in heads


def test_relative_clause_is_candidate():
    doc = parse("The report that arrived late was incomplete.")
    cands = enumerate_candidates(doc)
    assert any("that arrived late" in c.text for c in cands)


def test_adverb_is_candidate():
    doc = parse("She quickly finished the task.")
    cands = enumerate_candidates(doc)
    assert any(c.text == "quickly" for c in cands)


def test_candidate_char_ranges_slice_to_text():
    doc = parse("The committee approved the budget in the morning after long debate.")
    cands = enumerate_candidates(doc)
    assert cands  # non-empty
    for c in cands:
        assert doc.text[c.char_start:c.char_end] == c.text
        assert c.n_tokens >= 1


def test_enumeration_is_deterministic_and_sorted():
    doc = parse("The committee approved the budget in the morning after long debate.")
    a = enumerate_candidates(doc)
    b = enumerate_candidates(doc)
    assert a == b
    assert a == sorted(a, key=lambda c: (c.char_start, c.char_end))


def test_safe_and_core_are_disjoint():
    assert SAFE_DEPRELS.isdisjoint(CORE_DEPRELS)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_parse.py -v`
Expected: FAIL — `ImportError: cannot import name 'enumerate_candidates'`.

- [ ] **Step 3: Write the implementation (append to parse.py)**

```python
# src/promptcomp/parse.py  (append; add this import near the top)
from .types import Candidate

# spaCy (ClearNLP) adjunct/modifier labels — deletable subtree roots.
SAFE_DEPRELS = frozenset({
    "advmod",    # adverbial modifier
    "advcl",     # adverbial clause
    "acl",       # clausal modifier of noun
    "relcl",     # relative clause modifier
    "appos",     # appositional modifier
    "prep",      # prepositional phrase (UD obl/nmod equivalent)
    "npadvmod",  # noun phrase as adverbial modifier
    "amod",      # adjectival modifier (low priority, but eligible)
    "nmod",      # nominal modifier
    "parataxis",
    "discourse",
})

# spaCy (ClearNLP) core-argument / structural labels — never candidates.
CORE_DEPRELS = frozenset({
    "nsubj", "nsubjpass", "csubj", "csubjpass",
    "dobj", "dative", "iobj",
    "ccomp", "xcomp",
    "ROOT", "aux", "auxpass", "cop",
    "pobj",  # object of a preposition — belongs to its prep's subtree
})


def enumerate_candidates(doc) -> list[Candidate]:
    """Enumerate deletable subtrees (whole subtrees only)."""
    candidates: list[Candidate] = []
    for token in doc:
        if token.dep_ not in SAFE_DEPRELS:
            continue
        subtree = list(token.subtree)
        char_start = min(t.idx for t in subtree)
        char_end = max(t.idx + len(t.text) for t in subtree)
        candidates.append(
            Candidate(
                deprel=token.dep_,
                char_start=char_start,
                char_end=char_end,
                head_text=token.text,
                text=doc.text[char_start:char_end],
                n_tokens=len(subtree),
            )
        )
    candidates.sort(key=lambda c: (c.char_start, c.char_end))
    return candidates
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_parse.py -v`
Expected: PASS. If `test_prep_phrase_is_candidate` or `test_relative_clause_is_candidate` fails, print `token.dep_` for the sentence (Task 3 Step 2) and correct `SAFE_DEPRELS` to match the real spaCy labels — this is the expected place to reconcile the label scheme.

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest -v`
Expected: PASS — no regression to existing tests; `compress()` behavior unchanged.

- [ ] **Step 6: Commit**

```bash
git add src/promptcomp/parse.py tests/test_parse.py
git commit -m "feat: enumerate deletable subtree candidates from the parse"
```

---

## Task 5: inspect.py --show-candidates view

**Files:**
- Modify: `scripts/inspect.py`
- Test: `tests/test_inspect.py`

**Interfaces:**
- Consumes: `segment` (existing), `parse`/`enumerate_candidates` (new), `BlockKind`.
- Produces: a `--show-candidates` flag on the CLI. When set, after the normal compression summary, the CLI parses each PROSE block and prints its enumerated candidates (deprel + offset-adjusted text). spaCy is imported lazily (only when the flag is used) so normal runs stay light.

- [ ] **Step 1: Write the failing test (append to tests/test_inspect.py)**

```python
# tests/test_inspect.py  (append)
def test_show_candidates_lists_prose_candidates(tmp_path, capsys):
    doc = tmp_path / "doc.txt"
    doc.write_text("The committee approved the budget in the morning.\n")
    rc = inspect_cli.main(
        ["--file", str(doc), "--show-candidates", "--min-tokens", "0",
         "--cache-path", str(tmp_path / "c.sqlite")]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "candidates" in out.lower()
    assert "in the morning" in out
```

(`inspect_cli` is the module already loaded at the top of `tests/test_inspect.py` via `importlib`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_inspect.py::test_show_candidates_lists_prose_candidates -v`
Expected: FAIL — `--show-candidates` is not a recognized argument (SystemExit from argparse) or the output lacks the candidates section.

- [ ] **Step 3: Add the flag and the candidate view to inspect.py**

In `scripts/inspect.py`, add the argument (near the other `parser.add_argument` calls):

```python
    parser.add_argument(
        "--show-candidates", action="store_true",
        help="Parse each PROSE block and list enumerated deletable candidates.",
    )
```

Then, after the block that prints warnings and before the `if args.out:` block, add:

```python
    if args.show_candidates:
        from promptcomp.segment import segment
        from promptcomp.types import BlockKind
        from promptcomp.parse import parse, enumerate_candidates

        print("candidates    :")
        for block in segment(text):
            if block.kind is not BlockKind.PROSE:
                continue
            doc = parse(block.text)
            for c in enumerate_candidates(doc):
                # offset-adjust into the original document
                start = block.start + c.char_start
                print(f"    [{c.deprel:9}] {start:>6}: {c.text!r}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_inspect.py -v`
Expected: PASS (existing inspect tests + the new one).

- [ ] **Step 5: Verify on the real sample document**

Run: `python scripts/inspect.py --file data/samples/memo1.txt --ratio 0.6 --show-candidates`
Expected: prints the normal summary followed by a `candidates :` section listing adjunct/PP/clause candidates by deprel. Sanity-check that subjects/objects/main verbs do NOT appear as candidates and that the split looks linguistically reasonable (this is the M4 visual-verification checkpoint).

- [ ] **Step 6: Run the full suite**

Run: `python -m pytest -v`
Expected: PASS (all tests green).

- [ ] **Step 7: Commit**

```bash
git add scripts/inspect.py tests/test_inspect.py
git commit -m "feat: add --show-candidates view to inspection CLI"
```

---

## Self-Review Notes

**Spec coverage (M4 + carryover):**
- M4 Stage 1 parse + candidate enumeration (deprel taxonomy, whole-subtree, arg/adjunct split) → Tasks 3–4. ✓
- "Inspect candidates visually before deleting anything" (spec M4) → Task 5 `--show-candidates`. ✓
- `Candidate(span, deprel, head_token, char_range, token_ids)` intent (spec §3 Stage 1) → Task 2 `Candidate` (char_start/char_end + head_text + text + n_tokens; a serializable shape replacing raw spaCy token refs). ✓
- Deferred Plan A packaging fix → Task 1. ✓

**Intentional scope boundaries (deferred):** protect-list veto and the full invariant suite are Plan B2 (M5); scoring, budget, actual deletion, and the grammaticality re-parse test are Plan B3 (M6). `compress()` is deliberately unchanged here — enumeration is non-destructive and only surfaced through `inspect.py`.

**spaCy label risk:** `SAFE_DEPRELS`/`CORE_DEPRELS` use spaCy's ClearNLP labels, not UD. Task 3 Step 2 dumps real labels to confirm, and Task 4's behavioral tests (PP/relcl/adverb are candidates; subject/object/root are not) fail loudly if a label is wrong — the reconciliation point is called out explicitly.

**Determinism & non-regression:** enumeration sorts by `(char_start, char_end)` with a determinism test; the full suite is re-run in Tasks 4 and 5 to confirm the existing 33 tests and unchanged `compress()` behavior.

**Placeholder scan:** no TBD/TODO; every code step contains complete code. The one non-code artifact is the diagnostic label-dump command (Task 3 Step 2), which is intentionally not committed.
