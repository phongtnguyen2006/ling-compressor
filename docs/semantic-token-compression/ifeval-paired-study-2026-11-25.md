# IFEval paired compression study — 2026-11-25

This study checks two different claims on the same prompts: complete-input token savings, and whether a model still follows the **original** instructions after lexical substitution. [Google IFEval](https://huggingface.co/datasets/google/IFEval) supplies 541 prompts with executable instruction checks. The [Google evaluator](https://github.com/google-research/google-research/tree/master/instruction_following_eval) scores generated responses against each original prompt's constraint list. A small local model run is a feasibility demonstration, not evidence about the intended production model.

## Frozen inputs and compression

The official `ifeval_input_data.jsonl` was downloaded from dataset revision `966cd89545d6b6acfd7638bc708b98261ca58e84` and its local SHA-256 is `6a85310ca8ce15eff755aa08a3a4ff931c7e273e7515ebb3c492ea85fd8288f2`. `eval.prepare_ifeval` validated all 541 records and froze `prompts.jsonl`, original `gold.jsonl`, and provenance under ignored `.cache/semantic-ifeval/frozen-541/`. A stable key hash assigned 127 prompts to development and 414 to held-out; no dataset row was changed for the scorer.

The 400k GloVe pilot compared unchanged (U), curated/WordNet (B0), vector ranking (B1), and vector discovery (B2) with the same `o200k_base` tokenizer and eight-candidate budget. Across all 541 prompts, the original text totals **24,653 tokens**. B0 and B1 each propose **28 saved tokens (0.114%)**; B2 proposes **60 (0.243%)**, or **32 more than B0**. B2 changes 48 prompts. These are unreviewed proposals. The held-out B2 incremental saving over B0 is 24 tokens on 414 prompts; the group-bootstrap interval for mean additional saved tokens per held-out prompt is approximately **[0.034, 0.087]**. This does not establish meaning preservation or task quality. The [full local decision report](../../.cache/semantic-ifeval/decision-541-400k/summary.md) is generated when the source assets are provisioned.

## Paired response test

The local feasibility model is [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) at revision `7ae557604adf67be50417f59c2c2f167def9a775`. Its weights stay in the local Hugging Face cache. Google evaluator source is pinned locally to google-research commit `3f540f420bfae7e6ce48365a68d0ae003feb2f87`; its dependencies and NLTK `punkt_tab` were provisioned outside Git. Generation is greedy with the model's chat template. Each response record contains the exact proposed prompt hash, model identity, decoding settings, chat-template input token count, and output token count.

The first six-development-prompt diagnostic used a 128-token generation cap and produced 11 capped responses out of 24, so it is retained only as raw feasibility evidence. A second six-development-prompt diagnostic used a 256-token cap. Three rows for one prompt were capped; the paired scorer excludes that entire prompt before computing quality. The selected prompts were chosen after inspecting development edits to exercise the workflow, so they are **not** a random or held-out quality estimate.

The official scorer evaluated the remaining **five complete prompt sets** (20 arm responses). Four of these five had a change proposed by at least one arm. Its changed-only slice uses those same four prompts for every arm, including unchanged U. The model's complete chat-template input was 488 tokens for U, 486 for B0 and B1, and 485 for B2. Thus B2 saved **3/488 input tokens (0.61%)** on this selected subset. Strict original-instruction pass counts were **2/5 for every arm** (2/4 on the common changed-only slice); loose counts were **3/5 for every arm** (3/4 changed-only). No paired prompt-level regression or recovery was observed in this tiny sample. The unchanged model passed only two strict prompts, so equal pass counts cannot establish that substitutions preserve intent. The scorer fixes `langdetect` seed to 0 for repeatable language checks and records that setting in the report. See the generated [local paired report](../../.cache/semantic-ifeval/paired-qwen-dev-6-short-reviewed-v3/summary.md) when assets are provisioned.

## Reproduce with local assets

Provision the official IFEval input, the pinned evaluator source, and a local
snapshot of the named Qwen model before running these commands. Keep these
downloaded assets under ignored `.cache/` paths. The dataset source is
`https://huggingface.co/datasets/google/IFEval/resolve/966cd89545d6b6acfd7638bc708b98261ca58e84/ifeval_input_data.jsonl`;
verify its SHA-256 above. The evaluator is the
`instruction_following_eval` directory at google-research commit
`3f540f420bfae7e6ce48365a68d0ae003feb2f87`. Install its Python and NLTK
dependencies locally. The model snapshot revision is given above. The sample
ID file contains, one per line, `ifeval-1098`, `ifeval-1508`, `ifeval-1592`,
`ifeval-2417`, `ifeval-2739`, and `ifeval-3224`. These were deliberately
selected from development after edit inspection, so use a newly frozen,
representative selection for any quality estimate.

```bash
python -m eval.prepare_ifeval \
  --source .cache/semantic-ifeval/source/ifeval_input_data.jsonl \
  --out .cache/semantic-ifeval/frozen-541

NLTK_DATA="$PWD/.cache/nltk_data" OPENBLAS_NUM_THREADS=1 python -m eval.lexical_pilot \
  --corpus .cache/semantic-ifeval/frozen-541/prompts.jsonl \
  --out .cache/semantic-ifeval/pilot-541-400k \
  --corpus-label 'google/IFEval 541 official prompts; Apache-2.0; deterministic local development/held-out split' \
  --arms u,b0,b1,b2 --encoding o200k_base --wordnet \
  --vectors .cache/semantic-real/glove-wiki-gigaword-50.txt \
  --vector-label 'Gensim mirror of Stanford GloVe Wiki+Gigaword 6B 50d; full 400000 source-order vectors retained' \
  --max-words 400000 --max-candidates 8 --embedding-slots 4

python -m eval.ifeval_generate \
  --run .cache/semantic-ifeval/pilot-541-400k \
  --model-path /absolute/path/to/pinned/local/model/snapshot \
  --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --model-revision 7ae557604adf67be50417f59c2c2f167def9a775 \
  --doc-ids .cache/semantic-ifeval/dev-six-short-ids.txt --split development \
  --max-new-tokens 256 --out .cache/semantic-ifeval/local-qwen-dev-6-short

NLTK_DATA="$PWD/.cache/nltk_data" \
PYTHONPATH="$PWD/.cache/semantic-ifeval/official:$PWD/.cache/semantic-ifeval/deps:$PYTHONPATH" \
python -m eval.ifeval_paired \
  --run .cache/semantic-ifeval/pilot-541-400k \
  --gold .cache/semantic-ifeval/frozen-541/gold.jsonl \
  --responses .cache/semantic-ifeval/local-qwen-dev-6-short/responses.jsonl \
  --generation-manifest .cache/semantic-ifeval/local-qwen-dev-6-short/manifest.json \
  --evaluator-path .cache/semantic-ifeval/official \
  --out .cache/semantic-ifeval/paired-qwen-dev-6-short-reproduction
```

Each output directory must be new. For a real claim, freeze a representative held-out workload and use the actual production model, its tokenizer and complete request template, fixed decoding settings, adequate generation limits, paired accuracy statistics, and independent human review of every changed prompt. Embedding cosine and BERTScore may support inspection but cannot certify instruction meaning or downstream performance.

[Project status](STATUS.md) · [Production gates](production-gates.md) · [Two-reviewer workflow](two-reviewer-study-2026-11-25.md)
