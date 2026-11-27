# Tree pruning and semantic substitution on 414 IFEval prompts

This paired study uses every prompt assigned to the repository's frozen IFEval held-out split. The 414 prompts contain 25 distinct instruction types. Earlier tree-pruning diagnostics already examined the first 20 of these prompts. The sample includes those 20 and uses one local instruction-following model.

Each original serves as its own control. The tree workflow runs with a target retained ratio of 0.9, phrase shortening disabled, and no minimum input length. The semantic workflow uses the frozen B2 proposals from the 400k GloVe pilot, including unreviewed edits. Both workflows process the original independently. We measured both the user text with `o200k_base` and the complete model input with Qwen's tokenizer and chat template.

We used Qwen2.5-0.5B-Instruct at revision `7ae557604adf67be50417f59c2c2f167def9a775`, greedy decoding in float16 with MLX-LM on Apple silicon, a 512-token response limit, and batches of up to 64 (32 concurrent completions). Identical prompt texts share one answer. Google's IFEval evaluator scores every answer against the original constraints. A pass requires satisfying every checked instruction. The paired report gives the original pass count, how many of those passes survived, new passes, and the effect of excluding pairs that reached the response limit. The confidence intervals measure variation across prompts, conditional on this model and decoding run.

## Findings

Of the 102 prompts Qwen passed originally, 72 still passed after tree pruning and 99 still passed after semantic substitution. Tree pruning gained 18 passes elsewhere. It reduced complete Qwen input by 6.42% while losing 30 baseline passes. Semantic substitution gained one pass elsewhere and reduced complete input by 0.13%. It changed 34 prompts, and among those it retained six of the nine original passes.

One AI reviewer inspected all 34 semantic edits at the prompt level. Three prompts were acceptable, ten were borderline, and 21 were rejected. For example, turning a request for a limerick into a request for “waterford” changes the task even when the answer can still satisfy an unrelated formatting check. The interpretation is specific to this model, public dataset, and output limit. A production estimate requires representative system prompts and the serving model. See the [edit-by-edit inspection](held-out-edit-review.md) and [414 paired score records](held-out-paired-scores.csv).


Token counts use o200k_base on the user prompt. A strict pass means the answer satisfied every original instruction checked by IFEval.

| Method | Prompts | Edited | Original tokens | Proposed tokens | Saved | Original answers capped | Method answers capped |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tree | 414 | 373 | 18,448 | 16,479 | 10.67% | 144 | 147 |
| semantic | 414 | 34 | 18,448 | 18,405 | 0.23% | 144 | 145 |

| Method | Qwen input tokens including chat template | Saved versus original |
| --- | ---: | ---: |
| original | 30,794 | 0 (0.00%) |
| tree | 28,816 | 1,978 (6.42%) |
| semantic | 30,754 | 40 (0.13%) |

## Strict instruction following

| Method | Sample | Pairs | Original passes | Method passes | Baseline passes retained | Lost | Gained | Change in percentage points [95% interval] |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tree | All | 414 | 102 | 90 | 72/102 | 30 | 18 | -2.90 [-8.22, +2.49] |
| tree | Edited | 373 | 89 | 77 | 59/89 | 30 | 18 | -3.22 [-9.10, +2.75] |
| tree | Both uncapped | 227 | 59 | 50 | 42/59 | 17 | 8 | -3.96 [-11.02, +3.27] |
| tree | Edited and both uncapped | 197 | 50 | 41 | 33/50 | 17 | 8 | -4.57 [-12.64, +3.73] |
| tree | All three uncapped | 225 | 59 | 50 | 42/59 | 17 | 8 | -4.00 [-11.11, +3.29] |
| semantic | All | 414 | 102 | 100 | 99/102 | 3 | 1 | -0.48 [-2.33, +1.42] |
| semantic | Edited | 34 | 9 | 7 | 6/9 | 3 | 1 | -5.88 [-25.90, +15.92] |
| semantic | Both uncapped | 267 | 69 | 67 | 67/69 | 2 | 0 | -0.75 [-3.01, +1.57] |
| semantic | Edited and both uncapped | 24 | 8 | 6 | 6/8 | 2 | 0 | -8.33 [-29.80, +15.98] |
| semantic | All three uncapped | 225 | 59 | 58 | 58/59 | 1 | 0 | -0.44 [-2.80, +1.92] |

## Loose instruction following

| Method | Sample | Pairs | Original passes | Method passes | Baseline passes retained | Lost | Gained | Change in percentage points [95% interval] |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tree | All | 414 | 109 | 97 | 77/109 | 32 | 20 | -2.90 [-8.42, +2.69] |
| tree | Edited | 373 | 96 | 84 | 64/96 | 32 | 20 | -3.22 [-9.32, +2.96] |
| tree | Both uncapped | 227 | 65 | 54 | 46/65 | 19 | 8 | -4.85 [-12.08, +2.60] |
| tree | Edited and both uncapped | 197 | 56 | 45 | 37/56 | 19 | 8 | -5.58 [-13.85, +2.96] |
| tree | All three uncapped | 225 | 65 | 54 | 46/65 | 19 | 8 | -4.89 [-12.19, +2.62] |
| semantic | All | 414 | 109 | 108 | 106/109 | 3 | 2 | -0.24 [-2.29, +1.83] |
| semantic | Edited | 34 | 9 | 8 | 6/9 | 3 | 2 | -2.94 [-25.43, +20.41] |
| semantic | Both uncapped | 267 | 75 | 74 | 73/75 | 2 | 1 | -0.37 [-3.00, +2.30] |
| semantic | Edited and both uncapped | 24 | 8 | 7 | 6/8 | 2 | 1 | -4.17 [-29.75, +23.08] |
| semantic | All three uncapped | 225 | 65 | 65 | 64/65 | 1 | 1 | +0.00 [-2.80, +2.80] |

Intervals combine two 97.5% exact binomial intervals for paired losses and gains using Bonferroni bounds. The all-prompt rows include answers that reached the generation cap. Each uncapped row excludes a pair if either answer reached the cap. Edited and pairwise uncapped subsets are specific to each method. Capping can itself depend on compression, so uncapped results are descriptive. Loose checks also allow specified formatting normalizations.

## Reproduce

Follow the [paired IFEval setup](ifeval-paired-study-2026-11-25.md) to provision the frozen prompts, the completed semantic pilot, the official evaluator and its dependencies, NLTK resources, and local model weights. The runner uses MLX-LM 0.32.2 and MLX 0.32.5 on Apple silicon. Install these versions in the evaluation environment alongside SciPy, the project's dependencies, and the spaCy English model. From the repository root, run:

```bash
export NLTK_DATA="$PWD/.cache/nltk_data"
export PYTHONPATH="$PWD/src:$PWD/.cache/semantic-ifeval/deps"
snapshot="$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
python -m eval.ifeval_workflows --snapshot "$snapshot" --out .cache/workflow-held-out-mlx
python -m eval.ifeval_workflow_report .cache/workflow-held-out-mlx/scores.json
python -m pytest -q -rx tests/test_compression_quality_regressions.py
```

The output directory stores raw answers in SQLite, the model settings, the evaluator hashes, and per-prompt scores. Rerunning the same generation command resumes completed batches. Use a new directory for a different model or decoding configuration. The two focused regression tests reproduce previously observed failures on an IFEval formatting instruction and a Dolly polarity reversal. The formatting test uses a synthetic variant of the previously inspected prompt.
