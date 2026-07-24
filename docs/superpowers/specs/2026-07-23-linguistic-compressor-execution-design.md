# Linguistic Prompt Compressor — Execution Design (Pass 1)

**Date:** 2026-07-23
**Status:** Approved — ready for implementation planning
**Source design:** [`COMPRESSOR_SPEC.md`](../../../COMPRESSOR_SPEC.md) (the authoritative product spec)

---

## Purpose of this document

`COMPRESSOR_SPEC.md` is a complete product design: it fixes the architecture (four
stages), the core API, the repo layout, the milestones (M1–M8), and the critical
invariant test. It also lists (§11) decisions that must not be relitigated.

This document does **not** re-decide any of that. It records the **execution
decisions** the spec deliberately left open, so implementation can proceed without
re-brainstorming them. Where this document and the spec agree, the spec wins; where
the spec is silent, this document is the decision of record.

**Scope of this execution:** Pass 1 milestones **M1–M7** (skeleton through the eval
harness). M8 (`LLMLinguaScorer` as our own scorer) is out of scope. Non-goals in
spec §2 remain non-goals.

---

## Decision 1 — Dependency strategy (optional-dependency extras)

The spec pulls in heavy ML dependencies. Making them all hard requirements would make
the repo painful to install and would block M1–M6 on downloads they do not need. We
split dependencies via `pyproject.toml` optional-dependency extras.

- **Core (always installed):** `spacy`, `pyyaml`, `hypothesis`, `pytest`, `tiktoken`.
  Sufficient for M1–M6 and the full invariant suite. The spaCy English model
  (`en_core_web_sm`) is the only model download core needs.
- **`[anthropic]` extra:** the Anthropic SDK. `AnthropicCounter` and the eval's
  model-graded steps import it **lazily** and raise a clear, actionable error
  ("install the `[anthropic]` extra and set `ANTHROPIC_API_KEY`") if used without it.
  `TiktokenCounter` is the default token counter everywhere.
- **`[llmlingua]` extra:** `llmlingua` + `torch`. Needed **only** by the eval's
  `llmlingua2` baseline — our own scorer is `HeuristicScorer` for all of M1–M7. If the
  package is absent, that baseline is **skipped with a logged warning**, never a crash.

**Net effect:** `pip install -e .` yields a working compressor and the full invariant
suite with no ML-model downloads beyond `en_core_web_sm`. Eval degrades gracefully to
whatever backends are installed.

## Decision 2 — Token counting default and gating

- `TiktokenCounter` (offline, free, fast) is the **default** counter for all local
  iteration and for structural eval metrics.
- `AnthropicCounter` is used only when an operation informs a cost/budget decision and
  is gated behind `ANTHROPIC_API_KEY` + the `[anthropic]` extra.
- Every reported token number records **which counter produced it** (per spec §6).
- Counts are cached by `sha256(text) + model` in a local sqlite file.
- The spec's caveat is encoded in code and README: Claude Opus 4.7+, Sonnet 5, Fable 5,
  and Mythos 5 use a newer tokenizer producing ~30% more tokens than earlier models;
  `tiktoken` is **not** a proxy for Claude for cost decisions.

## Decision 3 — Corpus and eval realism

- **Starter corpus, not 30 docs up front:** ship a small `data/samples/` set (3–5 long
  public-domain-style documents — e.g. a realistic contract/memo, a central-bank-
  statement-style text, a transcript excerpt) that exercises every stage and lets the
  invariant suite run for real. Provide `scripts/fetch_corpus.py` documenting/downloading
  public analogues (EDGAR filings, MeetingBank transcripts) to grow toward the ≥30 docs
  the spec wants for meaningful accuracy numbers.
- **Offline by default:** `none`, `stopword`, and `ours` baselines need no API;
  `llmlingua2` is skip-if-missing. The **model-graded** parts (gold-answer generation,
  LLM judge) are gated behind `ANTHROPIC_API_KEY`. Without a key the harness still
  computes `tokens_before/after`, `compression_ratio`, `protected_violation_count`,
  latency percentiles, and `parse_failure_rate`; accuracy metrics are reported as
  "skipped (no model)".
- **Gold answers are fixed** (spec §7.1): generated once to `eval/gold/<doc_id>.json`
  and never regenerated per run, to preserve comparability.

## Decision 4 — Determinism and error handling

- **Determinism (spec §11.4, a requirement):** `HeuristicScorer` is pure/CPU; spaCy
  parsing is deterministic. Candidates are ordered by `(score, char_start)` for stable
  tie-breaks. Every `CompressionResult` stamps `config_hash` and `protect_list_version`.
  `test_determinism.py` asserts byte-identical output across processes.
- **Graceful degradation:** a sentence that fails to parse is left **uncompressed**
  (never guessed at) and surfaced as a warning in the result and in `inspect.py` output.
  A missing spaCy model yields a clear install message. Passthrough blocks are copied
  byte-identical with preserved offsets so recombination is exact.

## Decision 5 — Repository setup

- `git init` in the current `ling-compressor/` directory.
- Package layout from spec §4 placed at the repo root (`src/promptcomp/`, `eval/`,
  `tests/`, `data/`, `scripts/`). `pyproject.toml` uses a src-layout package.
- This design document is committed as part of the initial history.

---

## Testing approach — the product is the test suite

Test-driven throughout. `tests/test_protect_invariants.py` is the build gate:

- **Hypothesis property test:** for any document × ratio, the multiset of protected
  tokens is identical before and after (spec §8).
- **Number multiset** identical before/after.
- **Negation count** non-decreasing per surviving sentence.
- **Passthrough integrity:** every PASSTHROUGH block byte-identical in output.
- **Grammaticality:** re-parsing each output sentence yields a `ROOT` with core
  arguments intact.
- **Idempotence:** `compress(compress(x))` does not delete much more.
- **Determinism:** `compress(x) == compress(x)` across processes.

A single violation fails the build. These run against the full local corpus.

---

## Plan decomposition

M1–M7 is delivered as three chained implementation plans, each independently
verifiable, each taken through the `writing-plans` skill in turn:

1. **Plan A (M1–M3):** `types.py`, `tokens.py` (both counters), `segment.py` +
   passthrough tests, `substitute.py` + standalone compression measurement,
   `scripts/inspect.py`. No spaCy required. Ends with a working lossless compressor.
2. **Plan B (M4–M6):** `parse.py` (spaCy candidate enumeration), `protect.py` + the
   full invariant suite, `scoring/heuristic.py` + budget, wired into
   `pipeline.compress()`. Ends with end-to-end deletion-based `compress()`.
3. **Plan C (M7):** eval harness — `tasks.py`, `baselines.py`, `metrics.py`,
   `report.py`, `harness.py`; offline-capable per Decision 3.

---

## Traceability to the spec

| Spec section | Covered by |
|---|---|
| §3 Architecture (Stages 0–4) | Plans A & B |
| §5 Core API / inspect harness | Plan A (inspect, types, tokens), Plan B (compress) |
| §6 Token counting | Plan A, Decision 2 |
| §7 Eval harness | Plan C, Decision 3 |
| §8 Invariant tests | Plan B, Testing approach |
| §9 Milestones M1–M7 | Plan decomposition |
| §11 Non-relitigated decisions | Honored as constraints throughout |
| §2 Non-goals, §12 open question | Untouched; out of scope for this execution |
