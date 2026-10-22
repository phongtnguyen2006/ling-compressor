---
id: SEM-RUN-f71a5481946c
type: experimental-evidence
status: unreviewed
tags: [semantic-compression, pilot]
---

# Lexical opportunity pilot

Corpus: Six authored integration examples; real WordNet; synthetic vectors; no human judgments

Tokenizer: `tiktoken:o200k_base`. Documents: 6.

Run fingerprint: `f71a5481946c18c2358ee9c3cc8b62244d1443534e44eaedcc83011b9a148896`

**These are unreviewed suggestions, not demonstrated semantic preservation.**

| Arm | Before | After | Saved | Reduction | Changed inputs | Word / phrase edits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| u | 68 | 68 | 0 | 0.00% | 0/6 | 0 / 0 |
| b0 | 68 | 62 | 6 | 8.82% | 2/6 | 0 / 3 |
| b1 | 68 | 62 | 6 | 8.82% | 2/6 | 0 / 3 |
| b2 | 68 | 62 | 6 | 8.82% | 2/6 | 0 / 3 |
| b2_union | 68 | 62 | 6 | 8.82% | 2/6 | 0 / 3 |

## Evidence

- [Manifest and versions](manifest.json)
- [Frozen inputs](inputs.jsonl)
- [Occurrence decisions and complete outputs](results.jsonl)
- [Baseline pools and offered candidates](candidate-pools.jsonl)
- [Summary data](summary.json)
- [Blind edit review](edit-review.jsonl)
- [Blind complete-prompt review](prompt-review.jsonl)
- [Attribution key — keep separate during review](review-key.jsonl)

Human judgment fields are intentionally blank. No annotation was inferred from cosine or token savings.

## Limitations

- No human judgments, downstream accuracy, semantic equivalence, or net API savings established.
- All-input token savings include unchanged documents; empty inputs contribute zero tokens.
- Latency is a single sequential pass, includes cold/warm variation, and is not an isolated performance benchmark.
- Candidate-pool headroom is not a universal compression bound.
- SYNTHETIC VECTOR FIXTURE: embedding results test software only, not GloVe quality.
