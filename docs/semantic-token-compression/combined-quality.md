# Combined tree pruning and semantic substitution on IFEval

The combined pipeline saved 1,514 of 30,794 complete Qwen input tokens (4.92%). Qwen passed every checked instruction on 97 of 414 combined prompts and 102 originals. Of the 102 original passes, 79 survived. The pipeline lost 23 passes and gained 18 elsewhere.

| Input | Prompt tokens (o200k_base) | Complete Qwen input tokens | Complete input saved | Strict passes | Loose passes | Answers reaching cap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original | 18,448 | 30,794 | — | 102 | 109 | 144 |
| Revised standalone tree | 16,622 | 28,957 | 5.97% | 96 | 104 | 150 |
| Combined pipeline after its tree stage | 16,974 | 29,311 | 4.82% | 100 | 109 | 151 |
| Combined pipeline after semantic edits | 16,943 | 29,280 | 4.92% | 97 | 106 | 148 |

The combined workflow discarded tree cuts on 56 prompts and used the original text for the next stage. This accounts for the different token counts between the two tree rows. Compared with standalone pruning, the final pipeline used 323 more complete-input tokens and gained one net strict pass (five gains, four losses). The paired 95% interval for the pass-rate difference was −2.37 to +2.84 percentage points.

The semantic step edited 27 prompts and saved 31 additional complete-input tokens across the dataset. Among these 27 cases, strict passes fell from eight to five. Three passes were lost and zero gained. All three regressions finished within the output limit in both stages. Across all 414 pairs the strict pass-rate change for this step was −0.72 percentage points (paired 95% interval −2.33 to +0.94). The measurements give little support for adding automatic vocabulary substitution to this tree workflow.

## Method and limitations

The study used the workbench compression handler at revision `74034a9`, the revised tree algorithm with a 0.9 retention target, and the B2 semantic arm with the full 400k GloVe index, WordNet, eight candidates, and four vector slots. All semantic proposals were evaluated automatically. Each intermediate tree output was checked against the revised standalone result or the original text on fallback.

All 414 prompts came from the same frozen split used to evaluate and refine tree pruning. Each output was graded against the original IFEval constraints with the same local Qwen2.5-0.5B-Instruct snapshot, greedy decoding, and 512-token response limit as the previous study. Identical inputs reused stored answers. The run generated answers for 25 previously unseen input texts and reproduced the recorded scores for every original and standalone tree control.

Among the 235 pairs where neither original nor combined generation reached the cap, 62 originals and 53 combined prompts passed strictly. IFEval checks the specified constraints. Factual correctness, human judgments of meaning, and performance on production system prompts remain unmeasured in this comparison. Intervals use the paired loss/gain method in the [baseline study](held-out-workflow-study.md).

[Per-prompt token counts, strict and loose scores, fallbacks, semantic changes, and cap flags](combined-paired-scores.csv)
