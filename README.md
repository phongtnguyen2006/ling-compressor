# promptcomp

A local, deterministic compressor for reducing the token cost of long-form English
prose. It removes low-value grammatical subtrees while preserving protected facts
such as numbers, negation, obligations, conditions, named entities, and defined
terms.

Unlike a generative summarizer, `promptcomp` does not rewrite the document or invent
new text. Its priority is an auditable safety boundary rather than the highest
possible compression ratio.

## How it works

```text
input
  → separate prose from code, tables, paths, and structured content
  → dependency-parse prose into complete grammatical subtrees
  → veto candidates containing or governed by protected language
  → rank the remaining candidates and delete the least informative
  → apply configured phrase shortening
  → recombine everything in its original order
```

Only complete, contiguous modifier subtrees can be deleted. Core arguments and
passthrough content are not candidates. Every result includes the deletions, vetoes,
protect-list version, and configuration hash that produced it.

The requested ratio is a target, not a promise. A protection-heavy document may
remain close to its original size because safety vetoes take precedence over the
token budget.

## Example compression

With `target_ratio=0.6` and `min_tokens=0`:

```text
Before (26 tokens)
The committee, after a long and unusually detailed discussion in the morning,
approved the annual budget in order to begin the project quickly.

After (17 tokens)
The committee, after a long and detailed discussion in the morning, approved the
budget.
```

The compressor removed `unusually`, `annual`, and
`in order to begin the project quickly`, saving 9 tokens (34.6%). The result's audit
trail records each deletion while the time reference remains protected.

## Quick start

Requires Python 3.12, spaCy, and the `en_core_web_sm` model.

```bash
python -m pip install -e ".[dev]"
python -m pip install "spacy>=3.7"
python -m spacy download en_core_web_sm
```

Start the visual workbench:

```bash
python scripts/ui.py
```

Then open [http://localhost:8000](http://localhost:8000). The website provides:

- bundled examples with highlighted deletions and protected spans;
- a free-form sentence and document workbench;
- token counts, timing, and a complete audit trail;
- an engineering diagram of the compression and safety paths.

The website permits short examples by disabling the library's normal 500-token size
gate for interactive requests.

## Python API

```python
from promptcomp import compress

result = compress(
    text,
    target_ratio=0.6,
    max_deletion_fraction=0.5,
)

print(result.compressed)
print(result.tokens_before, result.tokens_after)
print(result.deleted)  # what was removed, where, and why
print(result.vetoed)   # what was protected and by which rule
```

By default, inputs under 500 tokens are returned unchanged. Pass `min_tokens=0` when
you intentionally want to experiment with shorter text.

The terminal inspection tool exposes the same pipeline:

```bash
python scripts/inspect.py --file data/samples/memo1.txt --ratio 0.6
```

## Protected content

The hard-veto layer covers:

- numeric literals, money, percentages, quantities, and dates;
- negation and modal scope;
- obligations, conditions, exceptions, and quantifiers;
- people, organizations, places, laws, products, and other named entities;
- configured terms and terms extracted from definition patterns.

Over-protection is allowed; losing protected content is a test failure. The critical
invariants also cover deterministic output, passthrough integrity, contiguous subtree
deletion, and grammatical structure.

## Experimental LLMLingua hybrids

Install the optional backend with:

```bash
python -m pip install -e ".[llmlingua]"
```

[`eval/hybrid.py`](eval/hybrid.py) contains two research prototypes:

- `HybridCompressor` sends protected sentences through the conservative compressor
  and clean sentences through LLMLingua-2.
- `SentinelHybridCompressor` masks isolatable protected clauses, forces their
  sentinels and main-clause anchors through LLMLingua-2, restores the original clauses
  byte-for-byte, and falls back conservatively when validation fails.

```python
from eval.hybrid import SentinelHybridCompressor

compressor = SentinelHybridCompressor()
result = compressor.compress(text, 0.5)
print(result.text)
print(compressor.stats)
```

These hybrids compress further on some prose, but they are not part of the public API.
Protected-token validation is not the same as proof of semantic equivalence; use them
only for evaluation until downstream answer retention is measured.

## Evaluation and tests

Run the complete suite:

```bash
python -m pytest -q
```

Run the offline document × compressor × ratio grid:

```bash
python eval/run.py \
  --corpus data/samples \
  --out eval/results \
  --ratios 0.3,0.5,0.7
```

The grid compares uncompressed text, naive stopword removal, `promptcomp`, and
LLMLingua-2 when installed. `protected_violation_count` must remain zero for the
conservative compressor. The bundled corpus is deliberately small, so its results are
directional rather than production evidence.

Install `.[anthropic]` and set `ANTHROPIC_API_KEY` for model-graded accuracy. Use
`AnthropicCounter` for Claude cost decisions; the default tiktoken counter is only a
fast local estimate.

## Project map

```text
src/promptcomp/   compressor pipeline and packaged rule data
scripts/          inspection CLI, local web server, corpus helper
ui/               browser workbench
eval/             baselines, hybrid prototypes, metrics, reports
tests/            unit, property, invariant, and UI tests
data/samples/     small public-style evaluation corpus
```

See [COMPRESSOR_SPEC.md](COMPRESSOR_SPEC.md) for implementation details, guarantees,
known limitations, and the current evaluation summary.
