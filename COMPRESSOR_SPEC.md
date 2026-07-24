# Linguistic Prompt Compressor — Implementation Spec

**Status:** Pass 1 — standalone library + eval harness. No CLI, no proxy, no Claude Code integration.
**Audience:** Claude Code, implementing from scratch.
**Owner:** Sho

---

## 1. Goal

Build a **local, deterministic, non-generative** text compressor that reduces token count of
long-form prose while provably never deleting a protected class of tokens (numbers, negation,
conditionals, modals, named entities, defined terms).

The differentiator is **not** compression ratio. Off-the-shelf LLMLingua-2 will almost certainly
compress harder. The differentiator is an **auditable guarantee**: for any input, we can assert
that no protected token was removed, and every output sentence is grammatical because we only
delete whole dependency subtrees.

### What "done" means for Pass 1

1. `compress(text, ratio) -> CompressionResult` works on long-form prose input.
2. A harness lets me paste/feed a long document and inspect the output side by side.
3. Token counts (before/after) are computed with a real tokenizer.
4. An eval harness measures whether the compressed text still supports the same downstream answers.
5. A test suite asserts the protect-list invariant holds across a corpus. **Zero tolerance.**

---

## 2. Non-goals (explicitly out of scope for Pass 1)

These were evaluated and rejected. Do not build them.

| Not building | Why |
|---|---|
| CLI | Pass 2. Library first so eval is scriptable. |
| Claude Code hook / proxy integration | Pass 3. Interface is `pre_call`-shaped; see §11. |
| Compressing code | Research shows only ~10% compression on code before degradation. Code passes through untouched. |
| Compressing short instructions (<500 tokens) | ~27 tokens saved on a realistic prompt. Not worth the ambiguity risk. Size gate handles this. |
| Rewriting/abstractive compression | Generative methods hallucinate. Extractive only — we delete, never rewrite. |
| Article-stripping as a rule | Articles are single cheap tokens (~6-8% of text) and carry definiteness. Not a rule; let scoring decide. |
| Output-side compression | Info destroyed, restoration isn't programmatic. |

---

## 3. Architecture

Four stages. **The order matters: structure filters → protect-list vetoes → importance decides.**

```
input text
    │
    ├─ Stage 0: SEGMENT ──────────► pass-through blocks (code, tables, JSON, logs, paths)
    │                                        │
    ▼                                        │
  prose blocks                               │
    │                                        │
    ├─ Stage 1: PARSE + ENUMERATE            │
    │   dependency parse → candidate         │
    │   deletable subtrees (adjuncts,        │
    │   relative clauses, appositives, PPs)  │
    │                                        │
    ├─ Stage 2: PROTECT-LIST VETO            │
    │   strike any candidate containing a    │
    │   protected token. HARD RULE.          │
    │                                        │
    ├─ Stage 3: RANK + BUDGET                │
    │   score survivors by importance;       │
    │   delete cheapest until budget met     │
    │                                        │
    ├─ Stage 4: SUBSTITUTE (lossless)        │
    │   reversible phrase → short form       │
    │                                        │
    ▼                                        ▼
  recombine in original order ──────────────┘
    │
    ▼
CompressionResult
```

### Stage 0 — Segment

Split input into `PROSE` and `PASSTHROUGH` blocks. Never compress PASSTHROUGH.

Classify as PASSTHROUGH:
- fenced code blocks (``` ... ```) and indented code
- markdown tables
- JSON / YAML / XML blocks
- file paths, URLs, stack traces
- lines with >30% digits or >40% non-alpha characters
- lines matching a defined-term definition pattern (`"X" means ...`, `X ("the Y")`)

Everything else is PROSE. Emit `Block(kind, text, start, end)` preserving offsets so we can
reassemble exactly.

**This is the single highest-leverage stage.** Parsers fail badly on non-prose, and a wrong parse
means a wrong deletion.

### Stage 1 — Parse + enumerate candidates

Use spaCy (`en_core_web_sm` to start; benchmark `en_core_web_trf` later).

For each sentence, walk the dependency tree. A **candidate** is a whole subtree whose root has one
of these deprels (Universal Dependencies):

Safe-to-consider (adjuncts / modifiers):
- `advmod`, `advcl` — adverbial modifiers/clauses
- `acl`, `acl:relcl` — clausal modifiers, relative clauses
- `appos` — appositives
- `nmod`, `obl` — non-core nominal/oblique modifiers (often PPs)
- `amod` — adjectival modifiers (rank low priority; often meaningful)
- `discourse`, `parataxis`

Never candidates (core arguments — the argument/adjunct distinction does the work here):
- `nsubj`, `obj`, `iobj`, `ccomp`, `xcomp`, `root`, `aux`, `cop`

Emit `Candidate(span, deprel, head_token, char_range, token_ids)`.

**Invariant:** deleting a candidate must leave a linearizable, grammatical remnant. Deletions are
whole subtrees only — never partial spans. (This is the property LLMLingua-2 cannot guarantee,
because it scores tokens semi-independently and can slice through the middle of a constituent.)

### Stage 2 — Protect-list veto

A candidate is **struck permanently** if its span contains any of:

| Class | Detection |
|---|---|
| Numbers, currency, percent, dates | spaCy NER: `CARDINAL`, `MONEY`, `PERCENT`, `DATE`, `QUANTITY`, `ORDINAL` + regex |
| Negation | `neg` deprel; lemmas: not, no, never, without, nor, neither, none, cannot |
| Conditionals / exceptions | unless, provided that, except, subject to, contingent on, in the event, if, only if, otherwise |
| Modals / obligation | shall, must, may, will not, required to, entitled to, obligated, not to exceed, at least, no more than |
| Named entities | spaCy NER: `ORG`, `PERSON`, `GPE`, `LAW`, `FAC` |
| Defined terms | from `data/defined_terms.yaml` + auto-extracted from Stage 0 definition patterns |
| Scope-bearing quantifiers | all, each, any, every, solely, exclusively, at least, at most |

Also strike any candidate **whose span falls inside the scope of a negation or modal** — i.e. it is
a descendant of a token that has a `neg` child, or of a modal auxiliary. Scope-sensitive deletion
is the failure mode that turns "fees not to exceed $2M" into an unbounded liability.

Protect-list lives in `data/protect_terms.yaml`, versioned, hot-reloadable, with a `version` field
that gets stamped into every `CompressionResult` for audit.

### Stage 3 — Rank + budget

Score each surviving candidate by **information cost of deleting it**. Two backends behind one
interface:

```python
class Scorer(Protocol):
    def score(self, text: str, spans: list[Span]) -> list[float]: ...
```

- `HeuristicScorer` (default, zero-dependency): TF-IDF/IDF-weighted content-word density,
  position, deprel prior. Fast, CPU-only, deterministic. **Start here.**
- `LLMLinguaScorer`: wrap `llmlingua`'s token-classification probabilities
  (`microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank`). BERT-sized, CPU-tolerable.
- `PerplexityScorer`: GPT-2-small surprisal via HF transformers. Reference implementation of the
  original LLMLingua idea; slowest.

Delete candidates in ascending score order until the token budget is met or candidates are
exhausted. **Never exceed a configurable `max_deletion_fraction` (default 0.5)** — a hard stop
that prevents runaway compression.

Determinism requirement: same input + same config + same protect-list version → byte-identical
output. Seed everything; no sampling. This matters for cache-safety later.

### Stage 4 — Substitution (lossless, reversible)

Separate pass, independently toggleable. Dictionary lookup only, no model.

Two dictionaries in `data/substitutions.yaml`:
1. **Phrase shortening** (safe, one-way): `in order to → to`, `due to the fact that → because`,
   `at this point in time → now`, `for the purpose of → for`.
2. **Entity abbreviation** (bijective, reversible): repeated defined terms/long identifiers →
   short forms, with the mapping emitted in `CompressionResult.substitutions` so it can be
   inverted or supplied to the model as a legend.

Substitution is strictly better than deletion where it applies: **nothing is destroyed**, so it
cannot corrupt meaning. Build this early — it may deliver most of the win on entity-heavy
documents at a fraction of the risk.

---

## 4. Repo layout

```
promptcomp/
├── pyproject.toml
├── README.md
├── src/promptcomp/
│   ├── __init__.py
│   ├── types.py          # Block, Candidate, CompressionResult, Span
│   ├── segment.py        # Stage 0
│   ├── parse.py          # Stage 1
│   ├── protect.py        # Stage 2
│   ├── scoring/
│   │   ├── base.py       # Scorer protocol
│   │   ├── heuristic.py
│   │   ├── llmlingua.py
│   │   └── perplexity.py
│   ├── substitute.py     # Stage 4
│   ├── pipeline.py       # orchestrator: compress()
│   └── tokens.py         # token counting
├── eval/
│   ├── harness.py        # run compressors × documents × ratios
│   ├── tasks.py          # QA task generation + grading
│   ├── baselines.py      # none / stopword / llmlingua / ours
│   ├── metrics.py
│   └── report.py         # markdown + CSV output
├── tests/
│   ├── test_protect_invariants.py   # THE critical test
│   ├── test_segment.py
│   ├── test_determinism.py
│   └── test_grammaticality.py
├── data/
│   ├── protect_terms.yaml
│   ├── substitutions.yaml
│   ├── defined_terms.yaml
│   └── samples/          # long-form test documents
└── scripts/
    └── inspect.py        # paste text in, see diff out
```

---

## 5. Core API

```python
@dataclass(frozen=True)
class CompressionResult:
    original: str
    compressed: str
    tokens_before: int
    tokens_after: int
    ratio: float                      # tokens_after / tokens_before
    deleted: list[Deletion]           # span, text, deprel, score, reason
    vetoed: list[Veto]                # span, text, protect_class  <-- audit trail
    substitutions: dict[str, str]
    protect_list_version: str
    config_hash: str
    timings_ms: dict[str, float]

def compress(
    text: str,
    *,
    target_ratio: float = 0.6,
    scorer: Scorer | None = None,
    protect: ProtectList | None = None,
    substitute: bool = True,
    min_tokens: int = 500,            # size gate: below this, return unchanged
    max_deletion_fraction: float = 0.5,
) -> CompressionResult: ...
```

`vetoed` is not debug output — it is the **audit artifact**. It is the evidence that the protect-list
ran and what it caught. Persist it.

### Inspection harness (`scripts/inspect.py`)

Requirement: "put in long-form text and check the output."

```
python scripts/inspect.py --file data/samples/memo1.txt --ratio 0.6 --scorer heuristic
```

Prints:
- side-by-side or unified diff, **deletions colored by the deprel that licensed them**
- vetoed candidates in a separate color (what we *chose not* to cut, and why)
- token counts before/after, per tokenizer
- per-stage timing
- warnings: any sentence that failed to parse, any block classified PASSTHROUGH

Also support `--stdin` and `--out result.json`.

---

## 6. Token counting (`tokens.py`)

Two backends. **Do not use tiktoken as a proxy for Claude.**

```python
class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...

class TiktokenCounter:   # o200k_base / cl100k_base — GPT models, offline, free, fast
class AnthropicCounter:  # client.messages.count_tokens(...).input_tokens — exact for Claude
```

Critical caveat to encode in the code and README: per Anthropic's docs, **Claude Opus 4.7 and
later, Claude Sonnet 5, Fable 5, and Mythos 5 use a newer tokenizer that produces roughly 30% more
tokens for the same input text than earlier models.** So:

- Always report *which* counter produced a number.
- Use `AnthropicCounter` for anything that informs a cost or budget decision.
- Use `TiktokenCounter` only for fast local iteration; cache results (the endpoint is rate-limited
  though free).
- The token counting endpoint is an estimate and may differ slightly from actual billed input.

Cache counts by `sha256(text) + model` in a local sqlite file to avoid re-calling.

---

## 7. Eval harness — does the compressed version still work?

This is the part that decides whether the project is real. Structure it as
**documents × compressors × ratios → task accuracy**.

### 7.1 Task generation

For each source document:
1. Use a strong model to generate N (default 20) **question + gold-answer** pairs **from the
   original document only**. Require short, extractive, verifiable answers.
2. Manually spot-check a sample of gold answers once per document; cache them to
   `eval/gold/<doc_id>.json` so the gold set is **fixed** across runs. Never regenerate gold
   answers per run — that destroys comparability.
3. Bias question generation toward the categories the protect-list guards: amounts, dates,
   conditions, obligations, exceptions, parties. Those are the questions that expose corruption.

### 7.2 Task execution

For each (document, compressor, ratio):
- Build the prompt: `{compressed_context}\n\nQuestion: {q}\nAnswer concisely.`
- Call the target model at temperature 0.
- Grade the answer against gold with (a) exact/normalized match, (b) F1 over tokens, and
  (c) an LLM judge for semantic equivalence. Report all three; they disagree in informative ways.

### 7.3 Baselines (all four, every run)

| Baseline | Purpose |
|---|---|
| `none` | Ceiling. Accuracy with the full document. |
| `stopword` | Floor. Naive stopword removal. Should be visibly bad. |
| `llmlingua2` | The real competitor. Expect it to compress harder. |
| `ours` | The hybrid. |

### 7.4 Metrics

Report per (compressor, ratio):
- `tokens_before`, `tokens_after`, `compression_ratio`
- `accuracy_exact`, `accuracy_f1`, `accuracy_judge`
- **`accuracy_retention` = accuracy(compressor) / accuracy(none)** — the headline number
- **`protected_violation_count`** — must be 0 for `ours`. Any nonzero value is a build failure.
- `latency_ms_p50`, `latency_ms_p95` (compression only, excluding the model call)
- `parse_failure_rate`

Plot accuracy_retention vs compression_ratio for all four. The claim to validate is:
**at equal accuracy retention, `ours` compresses somewhat less than `llmlingua2` but has zero
protected violations, while `llmlingua2` has some.** If `llmlingua2` also has zero violations on
your corpus, the differentiator is weaker than assumed — that is a finding worth knowing early.

### 7.5 Corpus

Need ≥30 long-form documents (2k–20k tokens each) for meaningful numbers. Start with public,
non-sensitive analogues: SEC filings (EDGAR), earnings call transcripts, central bank statements,
MeetingBank transcripts. **Do not put real internal memos in the eval corpus** until the data
governance question in §12 is settled.

---

## 8. The critical test: protect-list invariants

`tests/test_protect_invariants.py`. This is the product.

```python
@given(document=st.sampled_from(CORPUS), ratio=st.floats(0.2, 0.9))
def test_no_protected_token_is_ever_deleted(document, ratio):
    result = compress(document, target_ratio=ratio)
    for cls, pattern in PROTECTED_CLASSES.items():
        before = extract(document, pattern)
        after = extract(result.compressed, pattern)
        assert before == after, f"{cls} token lost: {set(before) - set(after)}"
```

Additional invariants:
- **Numbers**: multiset of all numeric literals is identical before and after.
- **Negation**: count of negation tokens is non-decreasing per sentence that survives.
- **Determinism**: `compress(x) == compress(x)` across processes.
- **Idempotence check**: `compress(compress(x))` should not delete much more; large secondary
  deletion means Stage 3 is unstable.
- **Passthrough integrity**: every PASSTHROUGH block appears byte-identical in the output.
- **Grammaticality**: re-parse each output sentence; assert it has a `ROOT` and its core arguments
  are intact.

Run these on every commit against the full corpus. A single violation blocks the build.

---

## 9. Milestones

**M1 — Skeleton + tokens (day 1)**
`types.py`, `tokens.py` with both counters, `scripts/inspect.py` echoing input with token counts.

**M2 — Segmentation (day 1–2)**
Stage 0 with passthrough integrity tests. Verify on a document with code, tables, and prose mixed.

**M3 — Substitution pass (day 2)**
Stage 4 standalone. Measure how much it compresses *alone*. Cheap and safe — this may be a large
fraction of the total win.

**M4 — Parse + candidates (day 3–4)**
Stage 1. Inspect candidates visually before deleting anything. Verify the argument/adjunct split
looks right on real sentences.

**M5 — Protect-list (day 4–5)**
Stage 2 + the invariant test suite. Do not proceed until invariants pass.

**M6 — Heuristic scorer + budget (day 5–6)**
Stage 3 with `HeuristicScorer`. End-to-end `compress()` working.

**M7 — Eval harness (day 6–8)**
§7 in full, with all four baselines. **This is where you learn whether the project works.**

**M8 — LLMLingua scorer (day 8–9)**
Swap in `LLMLinguaScorer`, re-run eval, compare.

Ship M7 before optimizing anything.

---

## 10. Reference links

Give these to Claude Code to read. All verified.

### Directly relevant research

**PartPrompt — parse trees + entropy (closest prior art; read first)**
- Paper: https://arxiv.org/abs/2409.15395
- PDF: https://arxiv.org/pdf/2409.15395
- Code: https://github.com/LengendaryHippopotamus/PartPrompt
- Note: research code (`python -u main.py <dataset> <ratios> <hyperparams>`), not a library. Read
  it for the tree-scoring and root-ward/leaf-ward propagation algorithm; don't try to import it.

**LLMLingua family — the importance-scoring half**
- LLMLingua (EMNLP'23): https://arxiv.org/abs/2310.05736
- LongLLMLingua: https://arxiv.org/abs/2310.06839
- LLMLingua-2 (ACL'24 Findings): https://arxiv.org/abs/2403.12968
- LLMLingua-2 in ACL Anthology: https://aclanthology.org/2024.findings-acl.57
- Code: https://github.com/microsoft/LLMLingua
- Compressor class (read this for the API to wrap):
  https://github.com/microsoft/LLMLingua/blob/main/llmlingua/prompt_compressor.py
- Model: https://huggingface.co/microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank
- Project page: https://www.microsoft.com/en-us/research/project/llmlingua/llmlingua-2/

**Classical sentence compression — the structural half**
- Filippova & Strube 2008, *Dependency Tree Based Sentence Compression* (the subtree-pruning
  method): https://aclanthology.org/W08-1105/ · PDF: https://aclanthology.org/W08-1105.pdf
- Filippova & Strube 2008, *Sentence Fusion via Dependency Graph Compression*:
  https://aclanthology.org/D08-1019
- Human acceptability judgements for extractive sentence compression (clean modern description of
  the subtree-deletion framework): https://arxiv.org/pdf/1902.00489
- Query-focused Sentence Compression in Linear Time (has a reimplementation appendix with concrete
  engineering details): https://arxiv.org/pdf/1904.09051

**Why code is excluded**
- *Less is More: DocString Compression in Code Generation*: https://arxiv.org/pdf/2410.22793
  (~10% compression ceiling on code; docstring-targeted compression reaches 25–40%)

### Tooling

- spaCy dependency parsing: https://spacy.io/usage/linguistic-features
- spaCy models: https://spacy.io/models/en
- Universal Dependencies relations (the deprel taxonomy in §3):
  https://universaldependencies.org/u/dep/
- Stanza (alternative parser, more accurate, slower): https://stanfordnlp.github.io/stanza/
- tiktoken: https://github.com/openai/tiktoken
- Anthropic token counting: https://platform.claude.com/docs/en/build-with-claude/token-counting
- Hypothesis (property-based testing for §8): https://hypothesis.readthedocs.io/

### For Pass 3 (integration — do not build yet)

- LiteLLM prompt compression (the `pre_call` callback slot this plugs into):
  https://docs.litellm.ai/docs/completion/prompt_compression
- LiteLLM + Headroom proxy pattern: https://docs.litellm.ai/docs/proxy/headroom
- Kong AI Prompt Compressor (selective `<LLMLINGUA>` section tags — same idea as Stage 0):
  https://developer.konghq.com/plugins/ai-prompt-compressor/
- Claude Code context window & compaction: https://code.claude.com/docs/en/context-window

---

## 11. Design decisions worth not relitigating

1. **Structure filters, protect-list vetoes, importance decides — in that order.** Stacking these
   *reduces* what may be deleted; it does not increase compression. Compressing *less* than
   LLMLingua is the expected and correct outcome. If `ours` ever compresses *more* than
   `llmlingua2` at equal settings, a guardrail has leaked — treat it as a bug.

2. **Extractive only, whole subtrees only.** No generation, no rewriting, no partial-span deletion.
   Grammaticality comes by construction, not by luck.

3. **Substitution ≠ deletion.** Substitution is lossless and reversible and therefore always
   preferable where it applies. Prefer it.

4. **Determinism is a requirement, not a nicety.** Pass 3 appends compressed text into a cached
   context. Nondeterministic output would bust the prefix cache on every turn, which costs far
   more than compression saves.

5. **Size gate at 500 tokens.** Below it, return input unchanged. Short instructions are
   high-leverage and low-volume — the worst possible compression target.

6. **The protect-list is the product.** If the invariant tests don't pass, nothing else matters.

---

## 12. Open question to resolve before Pass 3

Before any real internal document flows through this to a cloud model — even compressed — confirm
with compliance/infosec whether that pipeline is approved (MNPI, client confidentiality,
retention). Running the compressor locally helps the posture (raw text never leaves the machine,
only the compressed version is sent) but does not by itself answer the question. Use public
analogues in the eval corpus until it is settled.
