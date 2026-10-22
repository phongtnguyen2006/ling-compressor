# Semantic embedding compression: project board

Branch: `feature/semantic-embedding-compression`. This is a separate experimental track from the existing deletion and hybrid workbench. The local suggestion engine does not run inside the default compressor.

## Outcome and acceptance

Provide an offline, auditable lexical substitution experiment and a visual demo. A run must identify its tokenizer, vector asset, candidate arm, complete-text token counts, selected edits, rejected candidates, and pending human review. Measured token reduction must be reported separately from semantic preservation. No paid API or model download occurs during a normal run.

| ID | Title / owner | Acceptance criterion | Depends on | Status |
| --- | --- | --- | --- | --- |
| T-00 | Guarded engine and pilot / Codex | Local vector lookup, B0/B1/B2 comparison, POS and protected-span guards, full-text token recount, audit export, focused tests | — | Complete (existing uncommitted pilot adopted on this branch) |
| T-01 | Streamlit dashboard / Codex (reassigned after Antigravity stalled) | Interactive text, arm and vector selection; before/after token and edit views; honest demo and price labels | T-00 | Complete (4 AppTest smoke checks and live local server) |
| T-02 | Benchmark reporter / Codex | Reproducible run summary; optional local semantic proxy; no invented quality or accuracy result | T-00 | Complete (10 focused tests and bundled-evidence CLI run) |
| T-03 | Packaging and usage / Orchestrator | Isolated extras, branch-specific entry points, mentor timeline, clear asset/setup instructions | T-01, T-02 | Complete (installable extras, README, usage, and timeline) |
| T-04 | Independent review and verification / Antigravity | Review another worker's code, fix findings, run relevant tests and a UI smoke check | T-01, T-02, T-03 | Complete; no concrete review findings, 542 tests passed, local server health passed |

## Fall study plan

| Weeks | Deliverable and check-in |
| --- | --- |
| 1–2 | Freeze tokenizers, representative prompt strata and metrics; provision a labeled GloVe file locally. Compare no edit, curated/WordNet, vector ranking, and vector discovery with equal candidate budgets. |
| 3 | Mentor check-in 1: approve corpus selection, vector provenance, review rubric, and stop/continue thresholds before measuring quality. |
| 4–5 | Analyze substitution coverage and token savings; inspect candidate and protected-span audits; refine only on development examples. |
| 6–7 | Evaluate POS and syntax guard rejection/acceptance patterns; conduct blinded edit and complete-prompt human review. |
| 8 | Mentor check-in 2: review the prototype, human judgments, changed-prompt harms, and measured runtime; decide whether contextual validation is justified. |
| 9–10 | Run held-out prompt sets and paired downstream tasks if a task definition and evaluator are available. Estimate API cost using actual token counts and user-provided prices. |
| 11 | Mentor check-in 3: present results, limitations, report, and release decision. Publish only assets and data whose licenses permit redistribution. |

The bundled vectors and six prompts are synthetic integration fixtures. They are suitable for software checks, not a semantic preservation benchmark. AG News can be prepared as a **local, preselected** JSONL manifest for the pilot; a topic classification result would require a separate paired classifier evaluation. Mentor reviews and human annotations remain future external work until actually completed.
