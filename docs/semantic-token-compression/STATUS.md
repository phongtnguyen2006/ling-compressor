---
id: SEM-STATUS-001
title: "Semantic Token Compression Project Status"
type: project-status
updated: 2026-11-27
status: active
tags:
  - ling-compressor
  - semantic-compression
  - progress
---

# Semantic token compression: project status

**Last checked:** 2026-11-27

**Branch:** `feature/semantic-embedding-compression` (pushed; separate from `main`)

**Current decision:** The local, reviewed-template workflow is available. The measured pilot does **not** support automatic vector-based rewriting.

## At a glance

| Area | Status | What that means |
| --- | --- | --- |
| Tokenization and vector lookup | Done | Local GloVe text vectors, exact cosine top-k lookup, 10k/50k/400k retained-row selection, WordNet candidates, and `tiktoken` counting are implemented. |
| Substitution engine | Done | B0 curated/WordNet, B1 vector ranking, and B2 vector discovery run with POS, protected-text, and full-prompt token checks. |
| Visual workbench | Done | Semantic substitution appears in the shared Ling Compressor UI with edit details, token counts, and local vector controls. |
| Combined tree and semantic path | Implemented | The opt-in path runs tree pruning, then semantic proposals on its output; stage counts and audits are shown separately. Final text needs review. |
| Reviewed exact templates | Done for local use | A person can accept or reject each selected edit, save an approved template, export its audit, and invalidate it when source or configuration changes. |
| Two-reviewer quality study | Ready for human work | A blinded 400k packet and local review screen support independent judgments; the paired reporter requires both reviewers before counting accepted savings. No human judgments have been collected. |
| Paired IFEval model evaluation | Completed on all 414 frozen held-out prompts | Local Qwen passed 102 originals, 90 tree-pruned prompts, and 100 semantic prompts under strict scoring. [Paired study and per-prompt records](held-out-workflow-study.md) |
| Public-data opportunity study | Partly done | Frozen Dolly and AG News proxy pilots and a paired report exist. Human fidelity review and an intended production workload are still missing. |
| Automatic production rewriting | No-go for tested setup | B2 had very small incremental savings and spot-inspected meaning changes. Promotion criteria have not been met. |

## Done

- Built a local, auditable substitution engine with provenance, rejected-candidate reasons, complete-text token recounting, and deterministic pilot exports.
- Consolidated the semantic demo into the existing Ling Compressor workbench. Run `python scripts/ui.py` and open `http://localhost:8000`; select **Semantic substitution** or the **Authored prompt template · B0** example.
- Added a local review workflow for fixed authored templates. Its approval API checks decisions against server-selected edits and validates saved templates against the live configuration.
- Ran frozen pilots with a real GloVe 50-dimensional asset on 400 Dolly prompts and 400 AG News articles. The original 10k Dolly run proposed **27 of 34,599 input tokens (0.078%)** saved, only **2 more** than B0. Across AG News, B2 proposed **15 of 20,423 input tokens (0.073%)** saved.
- Compared 10k, 50k, and 400k source-order GloVe retention on the same 400 Dolly prompts. Full-vocabulary B2 proposed **61 of 34,599 input tokens (0.176%)** saved, **36 more** than B0, with **33 of 400** prompts changed. These proposals remain unreviewed and some change meaning. See the [full-vocabulary sweep](full-vocabulary-sweep-2026-11-25.md).
- Prepared a [two-reviewer study](two-reviewer-study-2026-11-25.md) from the full 400k pilot: 41 unique changed outputs plus 20 unchanged controls in each blinded packet, a visual review screen, and a paired report. The blank-file smoke report records **0/41** jointly complete changed prompts and zero reviewed savings.
- Ran an AG News topic-classification proxy: original and B0/B1/B2 versions each scored **91%** on 200 held-out articles. This measures topic classification only.
- Froze 541 official IFEval prompts and original constraints (127 development, 414 held-out). A 400k GloVe B2 pilot proposed **60/24,653 tokens (0.243%)** saved, versus **28** for B0. In a hand-selected five-prompt local Qwen diagnostic, B2 saved **3/488 complete model-input tokens** and every arm passed **2/5** strict original-instruction checks. One capped prompt set was excluded. See the [paired IFEval study](ifeval-paired-study-2026-11-25.md); this does not establish held-out or production-model quality.
- Added an opt-in tree pruning plus semantic proposal path to the shared UI. On the first 20 frozen held-out IFEval prompts, a 0.9 tree target reduced **843 to 749** tokens, and the full 400k GloVe B2 stage proposed **2 additional tokens** saved, reaching **747**. Two tree outputs fell back when cuts overlapped frozen content. This is a small token diagnostic with no quality judgment; see the [combined mode guide](usage.md#combined-tree-pruning-and-semantic-proposals).
- The IFEval code passed an independent Antigravity review after fixes for deterministic language scoring, generation-arm subsets, baseline identity, frozen provenance, and truncation reporting. The full suite passed **694 tests**; the official scorer reproduced the five-prompt diagnostic.
- Completed independent code review and a live 400k browser demo. With the two-reviewer workflow added, the current full test suite passed **654 tests**. The review screen's JSONL export was validated against the frozen pilot.

## In progress / awaiting evidence

These are open research and product decisions; no agent or mentor activity is claimed to be underway today.

| Next item | Completion evidence |
| --- | --- |
| Review complete prompts and proposed edits | Two independent human reviewers judge changed prompts; disagreements and harmful flags are resolved. The full 400k pilot currently has **0 of 41** unique changed complete prompts judged by both reviewers. Candidate edit judgments remain separate. |
| Test the intended workload | Obtain a consented sample of repeated prompt templates, freeze development and held-out sets, and measure token savings and edit quality on those sets. Public datasets are proxies. |
| Measure downstream LLM outcomes | The local IFEval workflow is feasible. Next pin the intended model and tokenizer, sample a representative held-out task set, use adequate response limits, and compare unchanged and approved text with paired accuracy statistics. |
| Establish net value and runtime | Measure actual reuse counts, provider prices, review effort, caching, validation latency, failures, and recovery costs. The current token-price illustration is gross savings only. |
| Complete mentor check-ins | Review the selection, quality rubric, prototype findings, and release decision with the mentor. None of the three planned check-ins is recorded as complete. |

## Future ideas (not committed work)

1. Compare frequency-first retention with a task-targeted vocabulary or another trained vector source on the same frozen inputs; the full GloVe source-order sweep is complete.
2. Explore contextual phrase validation or a small model only if human review shows enough safe, token-saving opportunities to justify its cost.
3. Test subword-aware re-tokenization or a fine-tuned prompt compressor against the same held-out baselines and quality gates.
4. Add a deployment-ready reviewed-template service only after the workload, access controls, latency, and net economics are measured.

## Plan position

The Week 1–2 baseline and Week 4–5 engine are implemented. Week 6–7 syntax checks exist, while human review is open. Week 9–10 dashboard and public proxy evaluation include a 414-prompt paired IFEval study on local Qwen; representative held-out downstream testing remains open. Week 3, 8, and 11 mentor check-ins have not been recorded. The [project board](project-board.md) tracks completed engineering tickets; the [production gates](production-gates.md) define the remaining promotion evidence.

[Project index](README.md) · [Pilot results and decision](real-vector-pilot-2026-11-11.md) · [Usage guide](usage.md)
