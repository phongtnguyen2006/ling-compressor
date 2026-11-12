# Real-vector pilot: production decision

Date: 2026-11-11. Decision: **do not enable automatic vector-based prompt
rewriting**. Review-first reusable templates remain the viable limited workflow.
This is a pilot decision on two public proxy datasets, not a claim about every
embedding model or a completed production workload study.

## Frozen setup

| Item | Value |
| --- | --- |
| Tokenizer | `tiktoken:o200k_base` |
| Arms | U (unchanged), B0 (curated + WordNet), B1 (same baseline pool ranked with vectors), B2 (baseline slots plus vector neighbors) |
| Candidate budget | 8 per source; B2 reserves 4 vector slots |
| WordNet | Local version 3.0 |
| Trained vectors | [Gensim data mirror](https://github.com/piskvorky/gensim-data/blob/master/list.json) of [Stanford GloVe](https://nlp.stanford.edu/projects/glove/) Wiki+Gigaword 6B, 50 dimensions, PDDL; source has 400,000 entries, first 10,000 retained |
| Compressed mirror | MD5 `c289bc5d7f2f02c6dc9f2f9b67641813` (matches Gensim data catalog) |
| Decompressed vector file | SHA-256 `9a8a78e2c6da3115e97044cbcfb40736d430961410bf732a4e80776412be5cd3` |
| Prompt corpus | [Databricks Dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k), CC BY-SA 3.0: 25 development and 25 held-out prompts from each of 8 categories, 400 total; source SHA-256 `2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec` |
| Task proxy corpus | [AG News](https://huggingface.co/datasets/fancyzhx/ag_news), 200 development and 200 official-test articles balanced across 4 classes. The dataset card lists license as unknown; text kept local. |

The frozen local manifests and raw pilot artifacts are in ignored
`.cache/semantic-real/`, not distributed with this repository. Recreate them
with [the Dolly selector](../../eval/prepare_dolly.py) and [the AG News
selector](../../eval/prepare_ag_news.py), then run the commands in
[usage](usage.md). Selection was based on source text hashes before candidate
inspection. Both corpora are proxies for prompt work; neither is actual Ling
Compressor customer traffic. The first 10,000 source-order vectors are a
pruned vocabulary and may miss useful words.

Independent review led to development-first selection across all Dolly
categories and category-prefixed AG News family labels. The corrected Dolly
selection produced a byte-for-byte identical frozen `prompts.jsonl` (SHA-256
`364b14ce5d512defa3f68fd5094382ae24982ef4a672043ffdc42eaa66558198`),
so its original pilot run remains valid. AG News was rerun with the corrected
category labels before final reporting.

## Measured token opportunity

All percentages divide saved tokens by **all original input tokens**, including
unchanged prompts. The p95 figures are from one sequential pilot pass, not a
controlled warm latency benchmark. Complete prompt quality was not reviewed.

| Corpus / split | Original tokens | B0 saved | B1 saved | B2 saved | B2 extra over B1 | B2 p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dolly development (200) | 17,543 | 9 | 9 | 10 | 1 | 1,079 ms |
| Dolly held-out (200) | 17,056 | 16 | 16 | 17 | 1 | 866 ms |
| AG News development (200) | 10,056 | 3 | 3 | 4 | 1 | 237 ms |
| AG News held-out (200) | 10,367 | 9 | 9 | 11 | 2 | 475 ms |

Across all 400 Dolly prompts, B0 saved 25/34,599 tokens (0.072%); B1
matched B0; B2 saved 27/34,599 (0.078%). Across all 400 AG News articles,
B0 saved 12/20,423 (0.059%), B1 matched B0, and B2 saved 15/20,423
(0.073%). On Dolly held-out groups, B2 versus B1 gained a mean 0.005 tokens
per document, with a group-bootstrap 95% interval of **[-0.010, 0.020]**.
On AG News held-out groups the mean gain was 0.010, interval **[0, 0.025]**.
Neither supports a positive incremental value claim. [The paired report
generator](../../eval/semantic_decision.py) validates run hashes and paired
coverage and reports each category; its uncertainty is conditional on these
selected public examples.

For scale only: at a **hypothetical** $2 per million input tokens, all 27 B2
Dolly tokens saved would be $0.000054 of gross input cost for one pass over
400 prompts. If a million future requests had exactly the same saving rate,
the estimated gross saving would be $0.135; the two tokens of incremental B2
gain over B0 would contribute $0.01. That extrapolation is not a measured bill
or production forecast and excludes runtime, review, caching, and recovery.

## Failure inspection and interpretation

Spot inspection, not blinded human review, found meaning-changing vector
suggestions such as `sacked → resigned`, `shootout → sweep`, `Males → Adults`,
and `Elsewhere → Far`. One technical phrase became `de-id`, which also needs
domain review before use. These examples show why cosine proximity, POS
compatibility, and token savings cannot certify interchangeable wording.
Some WordNet proposals also warrant review; the B0 savings are unreviewed.
The existing review importer confirms **0 of 420 unique Dolly complete-prompt
review records completed** in the frozen pilot. It did not infer approval from
the software checks.

A separate [AG News classification proxy](../../eval/ag_news_accuracy.py)
trained a deterministic linear classifier on 20,000 official training articles
after excluding the selected evaluation texts and 7 ambiguous duplicate-text
hashes. On 200 official-test articles, original and B0/B1/B2 versions all
scored **91%** topic accuracy (0 paired percentage-point change). Only 7 B0
and 9 B2 held-out articles changed; topic accuracy cannot detect the
meaning-changing examples above. This is a task proxy, **not** LLM downstream
accuracy or a prompt-fidelity certificate. Raw predictions and the model
identity remain in ignored `.cache/semantic-real/ag-news-accuracy-v2/`.

The observed B2 improvement is far below the provisional one-percent
incremental saving gate in [production gates](production-gates.md), and the
held-out intervals include zero. The observed B2 pilot p95 also exceeds the
provisional 100 ms live budget, although that latency measurement needs a
dedicated repeatable benchmark. No independent human quality score, pinned
LLM task accuracy, provider bill, cache impact, or net cost result exists.
Automatic deployment therefore fails the evidence gate. A review-first
template feature can still let a person approve exact edits and reuse an
unchanged, version-bound approved output; its economics should be judged per
template and actual reuse volume.

For one authored placeholder template in the local workbench, a person can
approve `in order to → to` while preserving `{customer_name}` and `{order_id}`.
The complete template falls from 18 to 16 `o200k_base` tokens. A warm local
Python check of the saved template through `validate_and_apply` took a median
1.17 ms and p95 1.30 ms across 1,000 calls on this development machine. That
is an illustrative exact-template reuse path, not a network service latency
measurement or a semantic-quality judgment. The caller must supply the active,
trusted configuration identity; the local HTTP validation route recomputes it.

Pilot run fingerprints: Dolly `ca8ca1d3d822bab12a7b46332927944b27e2c65ad7ffdfa54c94c38a57e158c8`;
AG News `f3d86d06d6ea2186a41d931c9a8cceea2bc92895a3a1c64070226a4848e17f24`.
The reports and raw audit files remain local because the AG News license is
unspecified and large pretrained vectors should not be checked into Git.
