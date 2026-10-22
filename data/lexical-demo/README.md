---
id: SEM-FIXTURE-001
title: Lexical pilot integration fixtures
type: test-fixture
created: 2026-10-06
updated: 2026-10-07
tags: [semantic-compression, fixtures]
---

These six short English texts and twelve four-dimensional vectors were authored
for software integration checks. The vectors are synthetic numbers, not GloVe,
FastText, or learned semantic representations. The texts are not a representative
corpus, a held-out task benchmark, or human-reviewed evidence of meaning retention.

`prompts.jsonl` contains stable input IDs, family labels, and `split: smoke`.
`synthetic-vectors.txt` uses the word2vec text header format accepted by the loader.
The examples include phrase shortening, repeated occurrences, modal constraints,
exact strings/placeholders, vector proposals, and a concise unchanged input.
Zero accepted embedding edits is a valid outcome; do not weaken guards to obtain
a preferred demo result.

To exercise all arms explicitly:

```bash
python -m eval.lexical_pilot \
  --corpus data/lexical-demo/prompts.jsonl \
  --vectors data/lexical-demo/synthetic-vectors.txt \
  --synthetic-vectors \
  --vector-label "Authored synthetic fixture; not pretrained embeddings" \
  --corpus-label "Authored integration examples; not representative evaluation" \
  --arms u,b0,b1,b2,b2_union \
  --out .cache/lexical/synthetic-demo
```

Initialize the tokenizer explicitly as described in the project README before
running offline. To include WordNet, provision it explicitly and pass `--wordnet`;
the fixture flag still applies to the vectors. Output directories are not overwritten.

See the [implementation record](../../docs/semantic-token-compression/implementation.md)
for SEM-REQ-008, which requires synthetic provenance and unfilled review fields.
