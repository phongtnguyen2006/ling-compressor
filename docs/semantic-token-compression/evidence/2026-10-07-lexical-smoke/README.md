---
id: SEM-EVID-006
title: "Lexical Pilot Integration Smoke — 2026-10-07"
type: verification-record
created: 2026-10-07
updated: 2026-10-07
status: verified-software-only
tags: [semantic-compression, evidence, synthetic-fixture]
---

# Lexical pilot integration smoke

[Implementation record](../../implementation.md) · [Usage guide](../../usage.md) · [Generated result](summary.md)

This run verifies the pipeline using six authored examples, the installed real WordNet corpus, and **synthetic vectors**. It is not the representative opportunity study and provides no pretrained embedding or human semantic-quality evidence.

## Executed command (original location)

From the repository root, after provisioning assets as described in the usage guide:

```sh
NLTK_DATA="$PWD/.cache/nltk_data" python3 -m eval.lexical_pilot \
  --corpus data/lexical-demo/prompts.jsonl \
  --vectors data/lexical-demo/synthetic-vectors.txt \
  --synthetic-vectors --wordnet \
  --vector-label 'Authored synthetic test vectors; not pretrained embeddings' \
  --corpus-label 'Six authored integration examples; real WordNet; synthetic vectors; no human judgments' \
  --arms u,b0,b1,b2,b2_union \
  --out docs/evidence/2026-10-07-lexical-smoke
```

The saved run used the absolute repository path for `NLTK_DATA`; the equivalent setup above is portable. The run was subsequently moved from `docs/evidence/` to this project's `evidence/` folder during a documentation reorganization. The command records the original execution path; for a new run, choose a fresh path under `docs/semantic-token-compression/evidence/` or an ignored `eval/results/` directory. The runner refuses to overwrite evidence.

## Observed result

- Run fingerprint: `f71a5481946c18c2358ee9c3cc8b62244d1443534e44eaedcc83011b9a148896`.
- Six documents, 68 original `o200k_base` tokens. Unchanged arm: 68 → 68.
- B0, B1, B2, and B2 union: each 68 → 62 tokens, six tokens saved (8.82%), two changed documents, three phrase edits, zero word edits.
- Six unique edit-review subjects and eight unique prompt-review subjects were exported. All human judgments remain unanswered.
- The identical savings come from curated phrase edits. This run does **not** demonstrate incremental embedding value. One superficially plausible vector example is rejected by the grammar check; the fixture was not tuned to force vector savings.

The [manifest](manifest.json) records dependency, asset, input, configuration, and source-code fingerprints. [Checksums](checksums.json) cover the nine generated artifacts. This explanatory README was added afterward and is outside that generated checksum set. Timing fields describe one sequential smoke run, not a performance benchmark.

The source fingerprint records the implementation at execution time. Later additions to the review-reporting workflow do not retroactively change this frozen pilot run.
