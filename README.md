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
