# Preserving output instructions during tree pruning

The earlier pruner could shorten “Wrap the final answer with double quotes” by removing the required quotation format. The revised pruner protects modifiers governed by recognized imperative verbs that specify formatting or content, including “wrap,” “repeat,” and “include.” The audit labels these vetoes `output_instruction`. Descriptive sentences in the same prompt remain eligible for pruning.

| Measure across 414 IFEval prompts | Original | Previous tree | Revised tree |
| --- | ---: | ---: | ---: |
| Prompt tokens under o200k_base | 18,448 | 16,479 | 16,622 |
| Complete Qwen input tokens | 30,794 | 28,816 | 28,957 |
| Complete Qwen input saved | — | 6.42% | 5.97% |
| Answers passing every strict check | 102 | 90 | 96 |
| Original passes retained | 102 | 72 | 76 |
| Answers reaching the 512-token output limit | 144 | 147 | 150 |

The change affected 84 compressed prompts. Compared directly with the previous tree version, eight strict failures became passes and two passes became failures. Compared with the original prompts, the revised pruner lost 26 passes and gained 20. Prompt-only savings fell from 10.67% to 9.90%.

This is a regression comparison on the same 414 prompts that informed the change. It uses the original Qwen2.5-0.5B-Instruct snapshot, original answers, and official IFEval checks. Identical input texts reused existing answers. The run generated 73 new answers using the same greedy settings, backend and output limit. Results include capped answers. Generalization to fresh prompts or another serving model is unmeasured.

[Paired scores for all 414 prompts](output-directive-pairs.csv) · [Previous evaluation and setup](https://github.com/phongtnguyen2006/ling-compressor/blob/9dda9ee/docs/semantic-token-compression/held-out-workflow-study.md)

To reproduce from the integration branch with the original study's environment and cache:

```sh
python -m eval.ifeval_tree_revision --baseline .cache/workflow-held-out-mlx --out .cache/tree-directive-study
```
