---
id: SEM-INDEX-001
title: "Semantic Token Compression Documentation"
aliases:
  - Semantic Compression MOC
type: index
created: 2026-10-06
updated: 2026-11-25
status: active
tags:
  - ling-compressor
  - semantic-compression
  - MOC
---

# Semantic token compression

The offline substitution pilot and review reporter are implemented. The [implementation record](implementation.md) tracks requirements, decisions, test results, and the next research gate. A [real-vector public-proxy pilot](real-vector-pilot-2026-11-11.md) and [10k/50k/400k vocabulary sweep](full-vocabulary-sweep-2026-11-25.md) are complete; human review and evaluation on the intended production workload are still pending. The earlier six authored smoke inputs and synthetic vectors verify the software only.

The interactive semantic demo is available from the shared Ling Compressor workbench (`python scripts/ui.py`). Select **Semantic substitution** alongside the existing tree-pruning and guarded-model paths. The earlier standalone Streamlit prototype was retired when the interfaces were consolidated.

For a repeated authored prompt, the workbench now lets a person review each proposed edit, save an exact approved template, and export its audit. The [usage guide](usage.md#review-and-reuse-an-exact-prompt-template) describes the local workflow. Automatic vector substitution remains disabled by the [real-vector pilot decision](real-vector-pilot-2026-11-11.md).

| Read | Purpose |
| --- | --- |
| [Current status](STATUS.md) · SEM-STATUS-001 | What is done, what awaits evidence, and future ideas |
| [Feasibility report](feasibility.md) · SEM-RESEARCH-001 | Value, feasibility, prior work, risks, and semester timeline |
| [Innovator–critic plan](debate-plan.md) · SEM-PLAN-001 | Agreed experiments and decision gates |
| [Implementation record](implementation.md) · SEM-IMPL-001 | Requirements, decisions, milestones, and dated evidence |
| [Pilot usage](usage.md) · SEM-USAGE-001 | Setup, CLI/API, review rubric, and reproduction |
| [Project board and mentor timeline](project-board.md) | Branch-specific delivery tickets, verification gates, and three check-ins |
| [Real-vector pilot](real-vector-pilot-2026-11-11.md) | Dolly and AG News token results, failure inspection, and automatic-use decision |
| [Production gates](production-gates.md) | Provisional promotion thresholds and exact-template review boundary |

## Evidence

- [Pilot smoke run](evidence/2026-10-07-lexical-smoke/README.md) · SEM-EVID-006 — WordNet and synthetic vectors; 68 → 62 tokens through phrase edits, with no human judgments.
- [Review coverage](evidence/2026-10-07-review-coverage/README.md) · SEM-EVID-008 — 521 passing tests; all six edit and eight prompt reviews remain pending.

## Keeping records traceable

Use the stable `SEM-REQ`, `SEM-DEC`, `SEM-M`, and `SEM-EVID` IDs in the [implementation record](implementation.md). Keep source inputs, asset and tokenizer versions, raw outputs, and checksums with each experiment. Label synthetic fixtures, unreviewed results, and deferred work. The research notes retain their original September 24 dates; moving a file or updating navigation does not revalidate its literature claims.

The repository's `docs/semantic-token-compression/` folder is canonical Markdown. In the local `LingTokenOptimizer` vault, open **Ling Compressor → docs → semantic-token-compression → README**. The vault's `docs`, `src`, and `eval` entries are relative symlinks to the repository; the vault MOC links here. Obsidian opened and rendered the implementation note on 2026-10-07 (SEM-EVID-007). No vault settings are required. For another checkout, open the repository as a vault or recreate those relative bridges.
