---
id: SEM-IMPL-001
title: "Semantic Token Compression Implementation Record"
aliases:
  - Semantic Compression Implementation
type: implementation-record
created: 2026-10-06
updated: 2026-10-07
status: in-progress
tags:
  - ling-compressor
  - semantic-compression
  - implementation
  - traceability
---

# Semantic token compression implementation record

[Documentation index](README.md) · [Feasibility research](feasibility.md) · [Debate and agreed plan](debate-plan.md) · [Pilot usage](usage.md)

This record turns the September 24 research and debate into a traceable implementation. The October 6 instruction authorizes implementation and requests Markdown viewable in Obsidian. The first milestone builds an executable offline opportunity pilot with auditable substitutions. Completing the software does not complete the representative human study or establish semantic equivalence.

**Interface update, 2026-10-22:** the interactive semantic demo is now part of the existing Ling Compressor web workbench (`python scripts/ui.py`). The separate Streamlit prototype was removed so all three compression paths share one visual interface and launch command. This demo does not fulfill the later reviewed-template workflow or the human and downstream evaluation milestones below.

**Evidence update, 2026-11-11:** a [real GloVe public-proxy pilot](real-vector-pilot-2026-11-11.md) now compares U/B0/B1/B2 on frozen Dolly instruction prompts and AG News articles. B2 adds only 2 tokens beyond B1 across 400 Dolly prompts and has meaning-changing spot-inspected suggestions. Its held-out incremental interval crosses zero. This rejects automatic embedding-based rewriting for the tested configuration; human review, actual production prompts, pinned LLM task evaluation, and mentor check-ins remain open. The [provisional release gates](production-gates.md) state the required evidence.

## Scope and evidence boundary

English prose and an explicit local tokenizer are the starting scope. The experimental arms are unchanged input (`U`), curated rules plus WordNet (`B0`), the same candidate pool ranked using vectors (`B1`), and the baseline pool plus additional vector neighbors (`B2`). Phrase substitutions are labeled separately from word substitutions. The engine proposes occurrence-specific edits, protects structured/exact content, counts complete outputs, and exports judgments for later human review.

No automatic production rewriting, contextual-model quality claim, human-study result, AG News/SQuAD/IFEval accuracy result, paid model evaluation, or Streamlit delivery is implied by this milestone. Later milestones depend on measured incremental value, as required by the [debate plan](debate-plan.md).

## Stable requirements

These requirements trace to the user proposal, the feasibility report, and the debate plan. Status is deliberately split between software verification and research evidence.

| ID | Requirement and acceptance criterion | Source | Current status |
| --- | --- | --- | --- |
| SEM-REQ-001 | Keep canonical Markdown with dated provenance, stable IDs, relative backlinks, and a local Obsidian navigation path. Markdown links and bridge targets must resolve. | User instruction, 2026-10-06 | Verified on disk and in Obsidian; SEM-EVID-002/005/007 |
| SEM-REQ-002 | Provide an offline substitution-only API/CLI. It must not attribute deletion gains to lexical substitution or silently invoke paid endpoints. Missing external assets must be visible. | Debate sections 1–2 | Implemented; SEM-EVID-004/006 |
| SEM-REQ-003 | Make B0/B1 use the identical candidates before ranking; let B2 expose additional vector-derived candidates with explicit provenance. Support locally supplied GloVe-style text vectors and optional WordNet resources. | Debate section 1 | Implemented; SEM-EVID-004/006 |
| SEM-REQ-004 | Use common guards and an explicit candidate budget. Record B2 allocation, baseline/additional candidates, and any uncapped union diagnostic separately. Deterministic tie-breaking must make repeat runs comparable. | Debate section 1 | Implemented; SEM-EVID-004/006 |
| SEM-REQ-005 | Plan edits by original character occurrence, not global string replacement. Reject overlapping edits and protected ranges; preserve structured blocks, caller-frozen text, and applicable grammatical/case constraints. Initially allow at most one accepted edit per sentence. | Debate section 2; research section 2 | Implemented; SEM-EVID-004/005/006 |
| SEM-REQ-006 | Count original and complete edited strings under the named tokenizer. Require positive sequential savings for applied edits and positive complete-output savings, otherwise retain the original. Do not sum isolated word costs as achieved savings. | Debate sections 1–2 | Implemented; SEM-EVID-004/005/006 |
| SEM-REQ-007 | Export JSONL audits with original span/text, replacement, source, scores, decision/reason, and token counts/deltas. Record input/configuration/asset identities sufficient to reproduce the run. | Debate sections 1–2 | Implemented; SEM-EVID-004/006 |
| SEM-REQ-008 | Export review records for candidate edits and composed outputs. Leave human judgments unfilled unless a person actually reviewed them; identify fixture or synthetic data. | Debate sections 1 and 4 | Implemented; SEM-EVID-004/006 |
| SEM-REQ-009 | Report aggregate savings over all original tokens, including unchanged prompts; show changed-input coverage and per-arm counts. Label the workload, tokenizer, and any limitations beside results. | Debate section 1 | Implemented; SEM-EVID-004/006 |
| SEM-REQ-010 | Verify deterministic outputs, protected-content behavior, fair pools/budgets, nonpositive-gain fallback, malformed assets, and export/reproduction behavior. Keep existing compressor tests passing. | Debate hard invariants | Verified; SEM-EVID-004/008 |
| SEM-REQ-011 | Conduct the representative opportunity study: roughly 200 preselected prompts, stratified review of roughly 200–400 edits plus composed outputs, and a documented continue/stop decision. Pilot size estimates are not accuracy-certification criteria. | Debate section 1 | Partial: 400 frozen public-proxy prompts and a stop decision for automatic use; blinded human review and intended workload sample remain open |
| SEM-REQ-012 | Before automatic use, evaluate held-out downstream tasks, prompt-level harmful edits, contextual validation cost, and net economics with predefined thresholds and adequate statistical precision. | Debate sections 3–4 | Deferred; evidence gate |
| SEM-REQ-013 | Validate submitted annotations against a frozen run, reject changed review subjects and invalid judgments, retain pending/partial states, and report changed outputs separately from controls and candidate edits. | Debate sections 1 and 4; continuation, 2026-10-07 | Implemented and verified; SEM-EVID-008 |

## Decision records

### SEM-DEC-001 — Build the offline opportunity pilot first

- **Date:** 2026-10-06.
- **Status:** Accepted implementation scope.
- **Rationale:** [SEM-PLAN-001](debate-plan.md) requires incremental evidence before a broad runtime system. An executable harness and protected editing engine are prerequisites for collecting that evidence.
- **Consequence:** Deliver tooling and demonstration runs first; do not present those runs as completion of SEM-REQ-011.
- **Requirements:** SEM-REQ-002, SEM-REQ-009, SEM-REQ-011.

### SEM-DEC-002 — Keep ranking and candidate discovery separate

- **Date:** 2026-10-06.
- **Status:** Accepted.
- **Rationale:** Comparing arbitrary lexical and embedding pools confounds whether vectors improve ranking or discover new useful alternatives.
- **Consequence:** B0/B1 share candidates; B2 identifies extras and its budget allocation. Label word and phrase contributions separately. A wider union is a costlier diagnostic, not a fair-budget competitor.
- **Requirements:** SEM-REQ-003, SEM-REQ-004, SEM-REQ-009.

### SEM-DEC-003 — Protect each occurrence and recount complete text

- **Date:** 2026-10-06.
- **Status:** Accepted.
- **Rationale:** [SEM-RESEARCH-001](feasibility.md) identifies context-free global substitutions, antonym similarity, ambiguous senses, and token-boundary effects as hazards.
- **Consequence:** Preserve original offsets and protected regions, cap accepted edits per sentence, log abstention, and require complete-output token benefit. These invariants do not prove semantic equivalence.
- **Requirements:** SEM-REQ-005, SEM-REQ-006, SEM-REQ-010.

### SEM-DEC-004 — Use explicit local assets and reproducible evidence

- **Date:** 2026-10-06.
- **Status:** Accepted.
- **Rationale:** Hidden downloads, unversioned dictionaries, and fallback pools undermine an offline comparison and make results difficult to reproduce.
- **Consequence:** Record the selected tokenizer, available/missing WordNet resources, vector provenance, configuration, and input identity. Small synthetic vectors can exercise code but cannot establish GloVe's value.
- **Requirements:** SEM-REQ-002, SEM-REQ-003, SEM-REQ-007, SEM-REQ-008.

### SEM-DEC-005 — Canonical notes live in the repository

- **Date:** 2026-10-06.
- **Status:** Accepted.
- **Rationale:** The user requires traceable Markdown viewable in Obsidian, and the existing local vault already uses repository symlinks.
- **Consequence:** Keep research, decisions, implementation state, and evidence under `docs/`; expose that directory in the vault with relative links. Preserve historical research dates and the local vault's unrelated notes/settings.
- **Requirements:** SEM-REQ-001.

### SEM-DEC-006 — Keep annotation completion separate from software completion

- **Date:** 2026-10-07.
- **Status:** Accepted.
- **Rationale:** The pilot generates review forms, but reporting useful embedding opportunities requires actual judgments on fixed subjects. Existing local examples do not constitute the proposed representative dataset.
- **Consequence:** Add a strict local annotation importer and coverage report. Missing or partial answers remain unanswered; accepted candidate edits do not approve composed outputs. Do not invent reviewers or label model judgments as human review.
- **Requirements:** SEM-REQ-008, SEM-REQ-011, SEM-REQ-013.

## Milestones

| ID | Deliverable | Completion evidence | Status |
| --- | --- | --- | --- |
| SEM-M0 | Traceable documentation and Obsidian navigation | Metadata, ID tables, backlinks, filesystem link checks and opened note; SEM-EVID-002/005/007 | Complete |
| SEM-M1 | Executable offline substitution pilot | API/CLI, common protected engine, B0/B1/B2, deterministic JSONL audits/review export, focused tests, documented smoke run; SEM-EVID-004/006 | Complete |
| SEM-M2 | Representative opportunity study and decision | Frozen workload/splits, human judgments, complete-output measurements, incremental embedding analysis | In progress; public-proxy run measured, human and production-workload evidence open |
| SEM-M3 | One contextual validator experiment | Rules-only/model comparison; frozen development thresholds; runtime/memory evidence | Conditional on SEM-M2 |
| SEM-M4 | Held-out task and economics evaluation | Paired accuracy/failure metrics, intervals, explicit thresholds, cache/recovery costs | Conditional |
| SEM-M5 | Reviewed-template workflow in unified workbench | Inspect, accept/reject, recount, save, validate, and export exact authored templates with an audit; measured and estimated savings separated | In progress; implementation under independent approval-bypass review |
| SEM-M6 | Hybrid ordering experiment | Standalone versus combined pipelines; marginal attribution; renewed protections | Deferred |

## Implementation and verification evidence

Evidence entries record commands or checking methods, result summaries, affected source/test paths, and limitations after execution. Baseline tests and documentation checks do not establish semantic accuracy.

### Source and verification map

| Responsibility | Source | Requirements | Verification files |
| --- | --- | --- | --- |
| Candidate contract and public API | [Models](../../src/promptcomp/lexical/models.py), [exports](../../src/promptcomp/lexical/__init__.py) | SEM-REQ-002, SEM-REQ-007 | `tests/test_lexical_candidates.py`, `tests/test_lexical_engine.py` |
| Local vector/WordNet sources and B0/B1/B2 allocation | [Candidates](../../src/promptcomp/lexical/candidates.py) | SEM-REQ-003, SEM-REQ-004, SEM-REQ-007 | `tests/test_lexical_candidates.py` |
| Protected occurrences and full-text token benefit | [Engine](../../src/promptcomp/lexical/engine.py), [protection](../../src/promptcomp/lexical/protection.py) | SEM-REQ-005, SEM-REQ-006, SEM-REQ-007 | `tests/test_lexical_engine.py` |
| Explicit offline tokenizer initialization | [Tokenizer loader](../../src/promptcomp/lexical/tokenization.py) | SEM-REQ-002, SEM-REQ-006 | `tests/test_lexical_pilot.py`, `tests/test_lexical_engine.py` |
| Corpus selection, audit/review exports, no-overwrite evidence, aggregate reporting | [Pilot runner](../../eval/lexical_pilot.py) | SEM-REQ-007, SEM-REQ-008, SEM-REQ-009 | `tests/test_lexical_pilot.py` |
| Source-bound annotation validation and per-arm coverage | [Review reporter](../../eval/lexical_review.py) | SEM-REQ-008, SEM-REQ-011, SEM-REQ-013 | `tests/test_lexical_review.py` |
| Setup, usage, and review handoff | [SEM-USAGE-001](usage.md) | SEM-REQ-001, SEM-REQ-002, SEM-REQ-007, SEM-REQ-008 | SEM-EVID-005 |

All four new test files contribute to SEM-REQ-010. The `data/lexical-demo/` fixture contains six authored inputs and synthetic vectors for software checks, not the SEM-M2 study dataset.

### Verification entries

| Evidence ID | Date | What was checked | Result and requirements |
| --- | --- | --- | --- |
| SEM-EVID-001 | 2026-10-06 | Existing report dates, local MOC and repository-to-vault symlink convention | Established prior provenance and local integration scope; SEM-REQ-001 |
| SEM-EVID-002 | 2026-10-06 | Python `pathlib` check of relative Markdown targets in the four canonical notes and the same notes through the vault bridge; verify `docs`, `src`, and `eval` symlink targets | 54 local links checked; zero missing targets. Frontmatter present on all four notes. SEM-REQ-001; Obsidian GUI rendering not inspected |
| SEM-EVID-003 | 2026-10-06 | Pre-change baseline: `python3 -m pytest -q`, run from repository root by the coordinating implementation agent | 307 passed in 30.47 seconds. Establishes starting regression state for SEM-REQ-010; not verification of new lexical code |
| SEM-EVID-004 | 2026-10-07 | Full suite after implementation and review fixes: `python3 -m pytest -q`, run from repository root by the coordinating implementation agent | 470 passed in 21.18 seconds: 307 existing tests plus 163 lexical tests (45 candidate, 104 engine, 14 runner). Covers software contracts for SEM-REQ-002 through SEM-REQ-010; no semantic-quality inference |
| SEM-EVID-005 | 2026-10-07 | Compare CLI help with [SEM-USAGE-001](usage.md), run its curated-only API/protected-span example, and check relative links in five canonical Markdown notes and through the existing vault bridge | API example measures 9→7 `o200k_base` tokens with review pending; explicit protection retains the original. 40 repository links plus 40 vault links resolve; zero missing targets. SEM-REQ-001, SEM-REQ-002, SEM-REQ-005, SEM-REQ-006; no Obsidian GUI inspection |
| SEM-EVID-006 | 2026-10-07 | Run all five arms with local WordNet and authored synthetic vectors; save [command and raw artifacts](evidence/2026-10-07-lexical-smoke/README.md) | Six inputs, 68 original tokens; each compression arm saves six tokens from three phrase edits. Six edit and eight prompt reviews remain unanswered. Pipeline integration verified; no incremental embedding value established. SEM-REQ-002 through SEM-REQ-010 |
| SEM-EVID-007 | 2026-10-07 | Open this canonical note through the existing LingTokenOptimizer vault using Obsidian 1.14.4; inspect its live UI | Note title, frontmatter properties, navigation links, and eight backlinks are visible. Completes actual GUI check for SEM-REQ-001; historical filesystem-only checks above retain their original limits |
| SEM-EVID-008 | 2026-10-07 | Independently implemented review-workflow tests; run full regression suite and a [coverage-only report](evidence/2026-10-07-review-coverage/README.md) against the frozen smoke run | 521 tests passed in 19.44 seconds (307 existing plus 214 lexical). Six edit subjects and eight prompt subjects remain pending; no invented judgments or accepted-output savings. SEM-REQ-008/010/013 |
| SEM-EVID-009 | 2026-10-07 | Parse frontmatter and resolve Markdown links in nine project/evidence notes on disk and through the vault; verify generated checksums and reporter code identity; run `git diff --check` | 76 repository links plus 76 vault links resolve; no missing targets. Nine pilot artifacts and four review-report artifacts match checksums. Note IDs/status metadata are present, pending totals are six edits/eight prompts, and diff whitespace checks pass. SEM-REQ-001/007/008/013 |
| SEM-EVID-010 | 2026-10-07 | Group semantic compression notes and evidence under `docs/semantic-token-compression/`; repair repository and vault links; parse note IDs; recheck generated checksums and `git diff --check`; open the new index in Obsidian 1.14.4 | Twelve inspected Markdown notes, ten unique documentation IDs, 164 local links checked across repository and vault with zero missing targets. All nine pilot and four review artifact checksums still match. Obsidian shows the project folder, index properties, and linked reading table. SEM-REQ-001/007 |

The baseline environment reported Python 3.12.3, spaCy 3.8.14, `en_core_web_sm` 3.8.0, NumPy 2.3.2, NLTK 3.10.0, and tiktoken 0.13.0. The `o200k_base` encoding was available on 2026-10-06. The WordNet corpus was absent at this checkpoint; NLTK being installed does not imply its corpus is present. This updates the historical report's offline-tokenizer availability observation without rewriting its September 24 findings.

## Reproduction and review handoff

The [pilot usage guide](usage.md) provides supported API/CLI commands, explicit local asset setup, arm definitions, and output-file descriptions. SEM-EVID-006 identifies the verified SEM-M1 run. Run artifacts must separate:

- input manifest, dataset/source labels, and exact input identity;
- candidate pools with lexical/vector provenance and fixed per-arm budgets;
- accepted and rejected occurrence audits with complete-prompt counts;
- unfilled human annotation records for edits and composed outputs;
- summary statistics and limitations of the selected workload.

Keep raw artifacts referenced by stable paths and hashes when reporting experimental findings. Do not treat a successful CLI smoke run or a synthetic vector fixture as a human semantic review. Document unavailable resources and deferred experiments instead of filling in estimated outcomes.

## Dated work log

### 2026-10-06 — Documentation and pilot implementation started

- **Authorization:** User requested traceable Markdown viewable in Obsidian, followed by implementation with subagents as needed.
- **Traceability:** Added this record and the documentation index; preserved the 2026-09-24 research and debate text while adding YAML metadata, portable local links, and backlinks.
- **Scope:** Accepted SEM-DEC-001 through SEM-DEC-005. Work begins on SEM-M1; representative human review and downstream evaluation remain separate milestones.
- **Local navigation:** Expose canonical `docs/` in the existing vault and preserve the existing MOC/history. Source directory bridges support inherited code links. No `.obsidian` settings or unrelated personal notes are changed.
- **Verification:** SEM-EVID-002 confirms 54 resolvable local links across canonical and vault paths. SEM-EVID-003 records the pre-change 307-test baseline. New implementation verification remains pending.

### 2026-10-07 — Implementation resumed and usage documented

- **Authorization:** User requested continuation of the documentation and implementation work.
- **Traceability:** Added [SEM-USAGE-001](usage.md) and linked it from this record and the index. Historical September 24 research and October 6 evidence entries remain preserved.
- **Implementation scope:** The working tree contains the independent lexical API, candidate providers, protected occurrence editor, pilot CLI/exporter, tests, and authored smoke fixtures. Candidate/engine review fixes, 470 passing tests, and the canonical WordNet smoke run complete SEM-M1 (SEM-EVID-004/006).
- **Actual application check:** Opened this note in Obsidian; SEM-EVID-007 completes SEM-M0.
- **Continued work:** Added the annotation validation/reporting step under SEM-REQ-013 and SEM-DEC-006, with independent critical review and a separate test worker. SEM-EVID-008 records 521 passing tests and a source-bound coverage report with all judgments pending. No human judgments are supplied by implementation work.
- **Final traceability check:** SEM-EVID-009 verifies all current documentation links and generated evidence checksums. Changes remain in the local working tree; no Git commit or remote publication was performed.

### 2026-10-07 — Documentation grouped by project

- **Navigation:** Moved the four semantic compression notes, project index, and evidence under `docs/semantic-token-compression/`. Kept a short `docs/README.md` gateway and simplified the local vault MOC.
- **Provenance:** Preserved note IDs, historical dates, generated artifacts, and their checksums. The evidence READMEs retain their original execution paths and explain the later move. Source, usage, fixture, and vault links now point to the grouped location.
- **Verification:** SEM-EVID-010 records filesystem and Obsidian checks after the move. Earlier evidence entries describe their original point-in-time paths and results.
- **Research boundary:** SEM-M2 representative selection and human review are not complete. Synthetic fixture results cannot establish semantic fidelity or GloVe value.
