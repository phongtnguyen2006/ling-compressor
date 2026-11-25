# Full GloVe vocabulary sweep — 2026-11-25

The local workbench now offers 10,000, 50,000, or 400,000 retained rows from a user-provisioned GloVe text file. It displays the actual retained and source row counts. The bundled 12-row vector file remains a synthetic software fixture. The full 400,000-row, 50-dimensional GloVe asset stays under ignored `.cache/` and is not distributed with this repository.

## Frozen comparison

The three runs used the same 400 Dolly instruction/context proxy prompts, `o200k_base` tokenizer, WordNet setting, candidate budget (8 with 4 vector slots for B2), full source-vector SHA-256, input hashes, and development/held-out splits. The [size-sweep reporter](../../eval/vector_size_report.py) checks these invariants and refuses an existing output directory. Its local JSON and complete split tables are in `.cache/semantic-real/vector-size-sweep-dolly-400/` when the source assets are provisioned.

| Retained GloVe rows | B0 tokens saved | B1 tokens saved | B2 tokens saved | B2 changed prompts | B2 p95 per prompt |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10,000 | 25 | 25 | 27 | 18 / 400 | 280 ms |
| 50,000 | 25 | 26 | 50 | 31 / 400 | 465 ms |
| 400,000 | 25 | 26 | 61 | 33 / 400 | 1,052 ms |

The original prompts contain **34,599 tokens**. The full-vocabulary B2 result proposes 61 saved tokens (0.176% of all input tokens), or 36 more than B0. On the 200 held-out inputs, B2 proposes 32 saved tokens versus B0's 16. Sequential pilot timings include proposal generation and are not isolated serving benchmarks. The 400k 50d float32 matrix is about 80 MB before Python objects and process overhead. A separate local cold-load probe took 4.25 seconds and reached approximately 383 MB process resident memory; this is a single machine observation, not a workload memory benchmark. The pilot manifests did not record peak RSS, so the reporter leaves that field unmeasured.

These are **unreviewed substitutions**, not realized savings or semantic fidelity scores. Manual inspection of 400k proposals found meaning-changing examples such as “undermines” → “reinforces,” “firmer” → “weaker,” and “netball” → “volleyball.” The larger vocabulary improves proposal coverage but does not make automatic rewriting suitable for production. Human approval of exact authored templates remains the supported local workflow. A future production gate needs complete-prompt human judgments, representative customer prompts, paired task outcomes, and net economics.

## Reproduce

First provision the GloVe and Dolly inputs and create the frozen corpus as described in the [usage guide](usage.md#reproduce-the-real-vector-prompt-pilot-locally). Then run the same `eval.lexical_pilot` command for each `--max-words` value (10000, 50000, 400000), changing only the size-specific output directory and descriptive vector label. Set `OPENBLAS_NUM_THREADS=1` for comparable local execution. Supply the three completed run directories to:

```bash
python -m eval.vector_size_report \
  --run-10k .cache/semantic-real/pilot-dolly-glove-10k-400-v4 \
  --run-50k .cache/semantic-real/pilot-dolly-glove-50k-400 \
  --run-400k .cache/semantic-real/pilot-dolly-glove-400k-400 \
  --out .cache/semantic-real/vector-size-sweep-dolly-400
```

Use a new `--out` directory for a repeat. The named directories are local evidence paths, not committed data or required package files.
