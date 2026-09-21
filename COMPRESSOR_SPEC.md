# Ling Compressor: system design

## Purpose

`promptcomp` compresses long prose without generation or rewriting. Its primary
guarantee is that protected language is not deleted. Compression ratio is secondary.

The current Pass 1 implementation includes the library, inspection CLI, local web
workbench, test suite, and offline evaluation harness. It does not include a proxy
or Claude Code integration.

## Pipeline

```text
input
  → segment prose from code/tables/structured text
  → dependency-parse prose and enumerate removable subtrees
  → veto candidates containing or scoped by protected language
  → score remaining candidates and select deletions within budget
  → apply safe phrase substitutions
  → recombine blocks in their original order
```

Processing order is intentional: structure limits where compression may happen,
protection removes unsafe candidates, and scoring chooses only among the survivors.

### Segmentation

`segment.py` separates prose from passthrough content such as fenced code, tables,
JSON-like blocks, paths, URLs, and stack traces. Passthrough blocks must remain
byte-identical.

### Candidate enumeration

`parse.py` uses spaCy's `en_core_web_sm` dependency parse. It considers complete
modifier or adjunct subtrees, including adverbials, prepositional phrases,
appositives, relative clauses, and adjectival modifiers. Core arguments are not
candidates. `delete.py` rejects non-contiguous subtrees before deletion.

### Protection

`protect.py` vetoes candidates that overlap:

- numbers, money, percentages, and dates;
- negation and negation scope;
- modal or obligation language and its scope;
- named entities;
- conditionals, exceptions, and quantifiers from `protect_terms.yaml`;
- configured or automatically extracted defined terms.

Over-protection is acceptable; under-protection is a build failure. Protection data
is versioned and included in every result.

### Scoring and deletion

`HeuristicScorer` assigns deletion cost using dependency-label priors, content-word
density, corpus rarity, and position. Lower-cost candidates are removed first until
the target budget or `max_deletion_fraction` is reached. Stable ordering and pure
CPU scoring make identical inputs and configuration byte-deterministic.

### Substitution

`substitute.py` applies configured phrase shortening such as `in order to → to`.
Mappings are returned in the result so changes remain inspectable.

## Public API

```python
compress(
    text,
    *,
    target_ratio=0.6,
    scorer=None,
    substitute=True,
    min_tokens=500,
    max_deletion_fraction=0.5,
    counter=None,
    cache_path=None,
) -> CompressionResult
```

`CompressionResult` contains the original and compressed text, token counts, actual
ratio, deletions, vetoes, substitutions, configuration hash, protect-list version,
timing, and warnings.

The 500-token size gate protects short, high-leverage instructions. The web
workbench sets `min_tokens=0` so individual sentences can be explored.

## Safety invariants

The tests enforce:

- protected numeric, negation, and configured-term multisets are preserved;
- passthrough blocks are byte-identical;
- only aligned, contiguous dependency subtrees are deleted;
- output is deterministic and substantially idempotent;
- output sentences retain parse roots and required structure.

`tests/test_protect_invariants.py` is the critical build gate.

## Evaluation

`eval/run.py` evaluates documents across compression ratios with these baselines:

| Baseline | Role |
| --- | --- |
| `none` | Uncompressed ceiling |
| `stopword` | Naive floor |
| `ours` | Conservative compressor |
| `llmlingua2` | Stronger optional competitor |

The bundled three-document result shows zero protected violations and zero parse
failures for `ours`. On the long memo, `ours` retained about 87% of tokens;
LLMLingua-2 compressed further but removed protected language. These are directional
results, not production evidence: accuracy retention is unmeasured and the corpus is
too small. See `eval/results/report.md` for the current run.

`eval/hybrid.py` contains two experiments: a sentence router and a clause-level
sentinel router that restores protected subordinate clauses byte-for-byte and falls
back when validation fails. Neither is part of the default API.

## Repository map

```text
src/promptcomp/   compressor library and packaged rule data
scripts/          inspection CLI, local web server, corpus helper
ui/               browser workbench assets
eval/             baselines, metrics, harness, reports
tests/            unit, property, invariant, and UI-server tests
data/samples/     small public-style evaluation corpus
```

## Known limitations

- spaCy and `en_core_web_sm` are required at runtime but are not yet declared in
  `pyproject.toml`.
- Modal and negation scope can extend across coordinated clauses, causing safe but
  weak compression on obligation-heavy text.
- Deletion selection is greedy by raw score rather than optimized per token or by a
  tree-knapsack algorithm.
- Pipeline timing currently reports total time, not per-stage timing.
- Model-graded accuracy needs an API key and a substantially larger public corpus.
- The native LLMLingua scorer proposed for a later milestone is not implemented.

Before sending internal documents to any cloud model, resolve applicable data
governance, confidentiality, and retention requirements. Local compression alone
does not answer that question.

## Design constraints

- Extractive only: no generated rewrites.
- Whole contiguous subtrees only: no arbitrary token slicing.
- Protection is a hard veto, never a scoring preference.
- Safe substitutions are preferred to destructive deletion.
- Same input plus configuration must produce identical output.
- Code and structured content pass through unchanged.
