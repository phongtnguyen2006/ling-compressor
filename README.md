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
