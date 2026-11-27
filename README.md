# Ling Compressor

Ling Compressor shortens English prompts and shows what changed. Its local workbench displays the original text, the proposed output, token counts, and the decisions behind each edit.

## Quick start

From the repository root, use Python 3.12 and run:

```bash
python -m pip install -e '.[lexical,dev]'
python -m spacy download en_core_web_sm
python -c "import tiktoken; tiktoken.get_encoding('o200k_base')"
python scripts/ui.py
```

Open [the local workbench](http://localhost:8000) and choose an example to inspect its edits. A [recorded tree pruning demo](demo/ling-compressor-demo.mp4) is also available.

## Compression modes

| Workbench mode | What it does |
| --- | --- |
| Tree pruning | Removes optional grammatical branches and records each cut or protection decision. |
| Guarded LLMLingua-2 | Uses a model to shorten text and restores protected clauses. Requires the optional `llmlingua` extra. [Setup and example](docs/guarded-hybrid.md) |
| Semantic substitution | Suggests shorter words or phrases for review. B0 uses curated rules, B1 ranks candidates with vectors, and B2 searches for vector neighbors. |
| Tree + semantic | Prunes first, then proposes substitutions on the remaining text. Shows token savings for each stage. |

For trained embeddings, select B1 or B2 and supply a local vector file. The bundled vectors are synthetic demo data. See the [semantic usage guide](docs/semantic-token-compression/usage.md) for asset setup and review instructions.

## Python API

```python
from promptcomp import compress

result = compress("The team approved the plan with great enthusiasm.", min_tokens=0)
print(result.compressed)
print(result.tokens_before, result.tokens_after)
print(result.deleted, result.vetoed)
```

The library skips inputs shorter than 500 tokens by default. The workbench and this example set the minimum to zero.

## Development and evaluation

```bash
python -m pytest -q
python scripts/inspect.py --file data/samples/memo1.txt --ratio 0.6
python eval/run.py --corpus data/samples --out eval/results --ratios 0.3,0.5,0.7
```

The compressor lives in `src/promptcomp/`, the workbench in `ui/` and `scripts/ui.py`, and the evaluation tools in `eval/`. See the [compressor specification](COMPRESSOR_SPEC.md), [evaluation summary](eval/SUMMARY.md), and [semantic project status](docs/semantic-token-compression/STATUS.md) for design decisions and measured results.
