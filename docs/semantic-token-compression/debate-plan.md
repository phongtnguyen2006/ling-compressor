---
id: SEM-PLAN-001
title: "Semantic Token Compression Debate Plan"
type: decision-record
created: 2026-09-24
updated: 2026-10-06
status: historical
tags:
  - ling-compressor
  - semantic-compression
  - research
---

**Implementation plan following the two-agent debate**

September 24, 2026. Participants: `implementation_innovator`, tasked with ambitious proposals, and `implementation_critic`, tasked with challenging assumptions and complexity. Both reviewed the [feasibility report](feasibility.md), exchanged proposals and objections directly, and revised their positions. This document records the resulting plan; no compressor implementation was changed.

**Decision: proceed with a gated research prototype.** Establish whether embeddings add useful savings before building a broad runtime system. Use reviewed, reusable templates as the first product workflow. Automatic live rewriting remains conditional on held-out quality and cost evidence.

| Decision debated | Innovator's position | Critic's challenge | Resolution |
|---|---|---|---|
| Offline versus live processing | Compile candidate lists offline; consider richer context checks at runtime | Validation can cost more than a small token reduction saves | Favor compile-once templates; benchmark live validation separately |
| Where innovation belongs | Explore embedding retrieval and offline phrase discovery | Curated phrase savings do not establish embedding value | Separate candidate ranking, candidate generation, and phrase discovery into different experiments |
| First experiment | Quickly find promising replacements in a small corpus | Selecting favorable prompts or edits can inflate savings | Sample prompts first; use all-input denominators and report abstention |
| Contextual validation | Consider stronger semantic and grammatical signals | Multiple validators add latency, dependencies, and ambiguous attribution | Compare rules alone with one contextual model first |
| Quality claims | Use a small annotated pilot to guide thresholds | A pilot cannot substantiate rare-error or narrow accuracy claims | Use it to identify opportunities/failures; reserve promotion decisions for frozen held-out evaluation |
| Savings headroom | Screen token-saving candidates before expensive review | Candidate search is incomplete and edits interact | Report headroom only within the tested pool; recount and review composed outputs |
| Initial interface | Make accepted/rejected edits inspectable | A polished dashboard can obscure an unproven engine | Build the thin Streamlit interface after preliminary measurements |

**1. Start with a 3–5 working-day opportunity study.** This duration and the following sample sizes are planning estimates. Select approximately 200 representative development prompts before discovering candidates. Include concise text, verbose prose, repeated templates, and protected-heavy inputs. Split document/template families to limit leakage; reserve separate families for later testing.

Pin the tokenizer and version. For offline experiments the existing local counter can be used with its encoding explicitly recorded. Provider cost or context-limit claims must use the actual selected model's counting/usage behavior rather than a surrogate tokenizer.

Compare these arms:

| Arm | Candidate pool | What it isolates |
|---|---|---|
| U | Unchanged input | Original token and task-quality baseline |
| B0 | Fixed curated phrase rules plus WordNet candidates | Value available without embeddings |
| B1 | Exactly B0's candidates, ranked using embeddings | Whether embeddings improve selection |
| B2 | B0 plus embedding-generated candidates | Whether embeddings find useful additional replacements |
| P, optional | Separately mined or manually curated phrases | Phrase-discovery value, reported independently from embedding gains |

Use the same guards, selected-candidate limit, and validation budget where comparing algorithms. Predeclare B2's allocation between baseline and additional candidates so they do not silently displace each other. Also run an uncapped union diagnostic that retains all baseline candidates and explicitly reports its additional retrieval, annotation, and validation costs. This distinguishes candidate headroom from performance under a fixed runtime budget.

Reject non-saving candidates cheaply before contextual work, then independently review a stratified sample of approximately 200–400 edits. Stratify by source, word/phrase, POS, ambiguity, and token gain. Report stratified findings directly; use known sampling weights if estimating a corpus-wide rate. Review compatible combined outputs as well as individual substitutions. Do not sum isolated token deltas and call the result achievable compression.

The pilot outputs should be a frozen input manifest, candidate tables, judgments, failure examples, complete-output token counts, changed-input coverage, and the additional accepted opportunities attributable to embeddings. Measure savings over all original input tokens, including unchanged examples. Also show workload slices so averages do not conceal poor cases.

**Pilot decision:** continue the tested embedding approach only if its reviewed incremental opportunities plausibly justify the expected validation cost and engineering effort for the chosen workload. If they do not, record the negative result and retain the simpler lexical baseline. Failure of one vocabulary, top-k setting, or domain rejects that tested design; it does not establish that all embedding approaches are futile. This is a diagnostic of the tested candidate pool, not a universal upper bound.

**2. Build one shared engine for protected, traceable edits.** Current deletion protections do not automatically guard later substitutions. Refactor the reusable detection logic into explicit protected spans before introducing broader rewriting. Begin with a separate substitution-only API so experiments cannot confuse deletion savings with substitution gains.

Recommended responsibilities, with new filenames treated as proposals:

| Module area | Responsibility |
|---|---|
| Existing `protect.py` | Expose protected ranges usable by both deletion and substitution |
| Proposed `lexical/candidates.py` | Load versioned candidates and identify their provenance |
| Proposed `lexical/planner.py` | Select occurrence-specific, nonoverlapping proposals |
| Proposed `lexical/validate.py` | Run lexical/grammatical checks and an optional contextual validator |
| Proposed `lexical/apply.py` | Apply edits, maintain original offsets, recount, and roll back |
| Existing `types.py` | Add per-occurrence records without discarding the current result contract |
| Existing evaluation modules | Run paired variants and export raw observations |

Reuse the tokenizer abstraction, parsing, segmentation, and packaged rule data. Keep heavier model dependencies optional. There is no demonstrated need for a hosted vector database in this plan.

Each proposed edit should record its original span/text, replacement, candidate source, lexical constraints, contextual evidence, sequential token delta, and acceptance/rejection reason. Record model, tokenizer, lexicon, configuration versions, and final total token delta. Define per-edit deltas relative to the actual application order; they are not universal additive word costs.

Initially permit at most one edit per sentence to simplify attribution and debugging. Preserve capitalization and inflection, freeze explicitly protected spans, and maintain an uncompressed original. Apply a consistent edit order and stable tie-breaking. If a structural check fails or a completed output does not show positive savings under the selected counter, return the original. Exact text invariants do not prove that surrounding relations or meaning remain unchanged.

**3. Test whether one contextual model earns its cost.** Compare the same candidate pipeline with rules alone and with one contextual model. Select the model on development evidence; do not begin with a masked language model, sentence encoder, and entailment ensemble together.

Measure whether the model reduces harmful edits at the same useful coverage, or increases accepted savings at the same error level. Include warm/cold latency, memory, and batch throughput. Calibrate thresholds on development data, then freeze them. High similarity and entailment predictions remain fallible signals rather than semantic guarantees.

Keep user-visible suggestions available even if evidence is insufficient for automatic application. User review is a workflow affordance, not a substitute for reporting downstream behavior accurately.

**4. Establish the deployment case with held-out tasks and cost accounting.** Evaluate complete prompts as well as edit precision: several individually plausible substitutions can interact, and a single altered instruction can spoil the entire request.

Compare unchanged input, the lexical variants, and the existing compressor. Use matched quality or matched token budgets where possible. Add LLMLingua-2 if the evaluation budget allows. Include classification, answer-bearing passage QA, instruction checks, human semantic review, and project-specific edge cases. Score responses against original task requirements, not merely similarity to the uncompressed model's output.

Before final testing, choose these thresholds for the actual deployment:

- Minimum worthwhile incremental token savings over B0 and the existing compressor.
- Acceptable prompt-level harmful-edit rate and primary task-accuracy margin.
- Required statistical precision and an appropriate sample size.
- p95 runtime and memory budgets.
- Positive net cost benefit after validation, cache behavior, and recovery.

The earlier report's 5% savings and one-percentage-point accuracy margin are candidate targets, not agreed requirements. A small pilot cannot support either claim. Use paired confidence intervals in the final evaluation and call an inadequately precise result inconclusive.

For reviewed template compilation, include the one-time validation/review expense and the expected number of reuses. Cache the approved result under the exact original content and relevant versions/configuration. Recompile when the template or dependencies change; do not assume that unseen variable substitutions are safe because one filled template was approved. Keep placeholders and exact output contracts immutable unless separately evaluated.

For live rewriting, count validation on every uncached request and measure the impact of changed prefixes on provider cache reuse. A compile-once result does not establish that live rewriting is economical.

**5. Deliver the thin Streamlit workflow, then consider hybrid compression.** The first product interaction is: load a reusable template, inspect proposed changes, approve or reject them, compare counts under an explicit tokenizer, and export the approved text with its audit. Show measured versus estimated costs distinctly. Display similarity as a diagnostic, never as a percentage of guaranteed meaning retention.

After the substitution-only system meets its gates, compare substitution before deletion and deletion before substitution. Reparse and recheck protections as needed between stages. Report both total savings and the marginal contribution of substitution. Keep the uncompressed and standalone variants available for rollback and fair comparisons.

**Hard invariants and evidence gates serve different purposes.** The engine must preserve designated exact spans and structured blocks, avoid overlapping or untracked edits, provide reproducible audit data, count complete outputs, and fall back when checks fail. These can be tested as implementation requirements. Useful savings, semantic fidelity, task accuracy, acceptable latency, and positive economics require empirical evidence on a specified workload.

The debate ended with no substantive disagreement. The innovator retained two experiments worth exploring—compile-once contextual validation and separately attributed phrase discovery. The critic secured fair ablations, an early stopping decision, complete-prompt evaluation, and conditional progression to automatic use. The recommended next action is the opportunity pilot, followed by the smallest engine justified by its results.


## Traceability

This document preserves the research and decisions recorded on **2026-09-24**. Its repository observations describe that date; implementation status is maintained separately. The 2026-10-06 edit adds metadata and portable navigation only. Source-file line numbers are historical references, not live anchors.

- [Documentation index](README.md)
- [Source feasibility research](feasibility.md)
- [Current implementation requirements, decisions, and evidence](implementation.md)
