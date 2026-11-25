---
id: SEM-STATUS-001
title: "Semantic Token Compression Project Status"
type: project-status
updated: 2026-11-25
status: active
tags:
  - ling-compressor
  - semantic-compression
  - progress
---

# Semantic token compression: project status

**Last checked:** 2026-11-25

**Branch:** `feature/semantic-embedding-compression` (pushed; separate from `main`)

**Current decision:** The local, reviewed-template workflow is available. The measured pilot does **not** support automatic vector-based rewriting.

## At a glance

| Area | Status | What that means |
| --- | --- | --- |
| Tokenization and vector lookup | Done | Local GloVe text vectors, exact cosine lookup, WordNet candidates, and `tiktoken` counting are implemented. |
| Substitution engine | Done | B0 curated/WordNet, B1 vector ranking, and B2 vector discovery run with POS, protected-text, and full-prompt token checks. |
| Visual workbench | Done | Semantic substitution appears in the shared Ling Compressor UI with edit details, token counts, and local vector controls. |
| Reviewed exact templates | Done for local use | A person can accept or reject each selected edit, save an approved template, export its audit, and invalidate it when source or configuration changes. |
| Public-data opportunity study | Partly done | Frozen Dolly and AG News proxy pilots and a paired report exist. Human fidelity review and an intended production workload are still missing. |
| Automatic production rewriting | No-go for tested setup | B2 had very small incremental savings and spot-inspected meaning changes. Promotion criteria have not been met. |

## Done

- Built a local, auditable substitution engine with provenance, rejected-candidate reasons, complete-text token recounting, and deterministic pilot exports.
- Consolidated the semantic demo into the existing Ling Compressor workbench. Run `python scripts/ui.py` and open `http://localhost:8000`; select **Semantic substitution** or the **Authored prompt template · B0** example.
- Added a local review workflow for fixed authored templates. Its approval API checks decisions against server-selected edits and validates saved templates against the live configuration.
- Ran frozen pilots with a real GloVe 50-dimensional asset on 400 Dolly prompts and 400 AG News articles. Across Dolly, B2 saved **27 of 34,599 input tokens (0.078%)**, only **2 tokens more** than B0. Across AG News, B2 saved **15 of 20,423 input tokens (0.073%)**.
- Ran an AG News topic-classification proxy: original and B0/B1/B2 versions each scored **91%** on 200 held-out articles. This measures topic classification only.
- Completed independent code reviews and the local UI/API approval demo. The last recorded full test run passed **609 tests**; this page does not claim a fresh test run.

## In progress / awaiting evidence

These are open research and product decisions; no agent or mentor activity is claimed to be underway today.

| Next item | Completion evidence |
| --- | --- |
| Review complete prompts and proposed edits | Two independent human reviewers judge changed prompts; disagreements are resolved and harmful-change rates are reported. The frozen Dolly pilot currently has **0 of 420** complete-prompt review records filled. |
| Test the intended workload | Obtain a consented sample of repeated prompt templates, freeze development and held-out sets, and measure token savings and edit quality on those sets. Public datasets are proxies. |
| Measure downstream LLM outcomes | Define a task with gold outcomes, pin the model and tokenizer, then compare unchanged and approved text on paired held-out inputs. |
| Establish net value and runtime | Measure actual reuse counts, provider prices, review effort, caching, validation latency, failures, and recovery costs. The current token-price illustration is gross savings only. |
| Complete mentor check-ins | Review the selection, quality rubric, prototype findings, and release decision with the mentor. None of the three planned check-ins is recorded as complete. |

## Future ideas (not committed work)

1. Compare a full or better-targeted trained vocabulary with the pilot's first 10,000 GloVe entries, while holding candidate budgets and evaluation inputs fixed.
2. Explore contextual phrase validation or a small model only if human review shows enough safe, token-saving opportunities to justify its cost.
3. Test subword-aware re-tokenization or a fine-tuned prompt compressor against the same held-out baselines and quality gates.
4. Add a deployment-ready reviewed-template service only after the workload, access controls, latency, and net economics are measured.

## Plan position

The Week 1–2 baseline and Week 4–5 engine are implemented. Week 6–7 syntax checks exist, while human review is open. Week 9–10 dashboard and public proxy evaluation are implemented in part; downstream LLM testing remains open. Week 3, 8, and 11 mentor check-ins have not been recorded. The [project board](project-board.md) tracks completed engineering tickets; the [production gates](production-gates.md) define the remaining promotion evidence.

[Project index](README.md) · [Pilot results and decision](real-vector-pilot-2026-11-11.md) · [Usage guide](usage.md)
