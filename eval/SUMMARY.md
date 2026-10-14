# Bundled-corpus evaluation

This is a reproducible snapshot of `promptcomp`'s conservative (`ours`)
compressor on the three text files in `data/samples/`. The run used the default
tiktoken counter and target ratios `0.3,0.5,0.7`. All three targets produced the
same output for each document in this run.

| Document | Input tokens | Output tokens | Retained | Protected-content violations | Parse failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| `memo1` | 1,282 | 1,116 | 87.1% | 0 | 0 |
| `notice_snip` | 72 | 72 | 100.0% | 0 | 0 |
| `policy_snip` | 79 | 71 | 89.9% | 0 | 0 |
| **Total** | **1,433** | **1,259** | **87.9%** | **0** | **0** |

Run from the repository root after following the README setup:

```bash
python eval/run.py --corpus data/samples --out eval/results --ratios 0.3,0.5,0.7
```

The command writes `eval/results/report.md` and `report.csv` locally. That
directory is ignored because it contains generated output. The optional
LLMLingua-2 baseline is included only when its dependency is installed.

These numbers measure token retention and this harness's protected-content
checks. The corpus is small, and downstream answer accuracy has not been
measured. Do not treat zero recorded violations as proof of semantic
equivalence on new documents.
