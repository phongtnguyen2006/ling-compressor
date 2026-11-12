# Production-value decision gates

Date: 2026-11-11. Status: provisional engineering criteria, awaiting mentor and
product-owner approval. Frozen before the Dolly pilot result was inspected.
The earlier AG News run was exploratory and is not a deployment workload.

## Evaluated workload and boundaries

- Primary proxy: 400 Databricks Dolly instruction/context prompts, 25 development
  and 25 held-out per each of eight task categories. Exact prompt text and the
  third-party reference responses remain under ignored `.cache/semantic-real/`.
- Secondary proxy: 400 balanced AG News articles in a fixed classification
  request. Its dataset card lists the license as unknown, so text remains local.
- Both are public benchmark proxies, not actual Ling Compressor customer traffic.
  A production decision ultimately needs a consented sample of the intended
  application templates and its actual tokenizer/provider prices.
- Pretrained vector source: Gensim's PDDL-licensed mirror of Stanford GloVe
  Wiki+Gigaword 6B, 50 dimensions, 400,000 source entries; exact-cosine pilot
  retains the first 10,000 source-order entries. This tests a pruned vocabulary,
  not the full GloVe model or all embedding methods.

## Promotion criteria for automatic rewriting

All criteria must pass on a frozen held-out workload after development choices
are locked. Missing evidence counts as *inconclusive*, not as a pass.

1. B2 saves at least one additional percent of **all original input tokens**
   over B0; the group-bootstrap 95% interval for its incremental saving is
   strictly above zero. B1 and B2 are reported separately so ranking and
   discovery are not conflated.
2. Two independent reviewers assess each changed complete prompt. Any
   disagreement or suspected harmful change is adjudicated; pending prompts
   are excluded from approval. The upper 95% bound on harmful changed-prompt
   rate is below 1%, and no unreviewed edit is auto applied.
3. A pinned downstream model and task set have an upper 95% bound on paired
   accuracy loss below one percentage point against unchanged input. A
   classifier proxy is diagnostic; it cannot satisfy an LLM accuracy gate.
4. Warm p95 compression time is at most 100 ms per target prompt on the
   deployment host. Cold-load time, peak memory, validation time, errors and
   fallback rates are reported separately.
5. A workload-specific net cost estimate is positive after validation,
   review/amortization, cache effects, and recovery costs. Input-token price
   alone gives gross potential savings, not net value.

These are conservative provisional thresholds for this experiment, not an
industry standard or a promise of safety. The mentor should revise and sign
them before any formal promotion study.

## Limited reviewed-template use

An individual reusable template can be useful before the general gate passes:
show its occurrence edits, let a person accept or reject each one, recount the
final complete text, and export the exact approved version with an audit. A
saved approval is valid only for identical source content, tokenizer, vector
asset and engine configuration. Any mismatch falls back to the source. The
value of one template depends on its measured token saving, actual reuse count,
the provider's input price and the review effort; the UI should not call gross
token estimates a net saving.

## External work still required

Mentor check-ins, independent human annotations, a pinned LLM evaluation with
gold task outcomes, production workload sampling, and actual provider billing
are not produced by the automated pilot. The project board tracks them as open
evidence gates; no software test may mark them complete.
