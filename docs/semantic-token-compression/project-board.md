# Semantic embedding compression: project board

Branch: `feature/semantic-embedding-compression`. This is a separate experimental track from the existing deletion and hybrid workbench. The local suggestion engine does not run inside the default compressor.

## Outcome and acceptance

Provide an offline, auditable lexical substitution experiment and a visual demo. A run must identify its tokenizer, vector asset, candidate arm, complete-text token counts, selected edits, rejected candidates, and pending human review. Measured token reduction must be reported separately from semantic preservation. No paid API or model download occurs during a normal run.

| ID | Title / owner | Acceptance criterion | Depends on | Status |
| --- | --- | --- | --- | --- |
| T-00 | Guarded engine and pilot / Codex | Local vector lookup, B0/B1/B2 comparison, POS and protected-span guards, full-text token recount, audit export, focused tests | — | Complete (existing uncommitted pilot adopted on this branch) |
| T-01 | Streamlit dashboard / Codex (reassigned after Antigravity stalled) | Interactive text, arm and vector selection; before/after token and edit views; honest demo and price labels | T-00 | Complete historically; superseded by T-06 |
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

## Unified workbench follow-up

Outcome: `python scripts/ui.py` serves one Ling Compressor interface for tree pruning, guarded LLMLingua-2, and semantic substitution. The semantic path keeps its experimental labeling, local vector provenance, complete-text counts, and occurrence audit. The separate Streamlit dashboard is retired after the unified path is verified.

The unified screen reuses the existing paper `#f7f8f4`, ink `#18201d`, blue measure `#3867d6`, teal protection `#087f73`, and coral rejection `#e86a4a` tokens, along with the same masthead, example shelf, mode cards, ruler, comparison columns, and audit list. Semantic controls appear only when that path is selected. This preserves the existing product identity while making the new algorithm visible through its occurrence edits.

| ID | Title / owner | Acceptance criterion | Depends on | Status |
| --- | --- | --- | --- | --- |
| T-05 | Unified semantic API / Codex | `/api/compress` accepts `mode=semantic`; local B0/B1/B2 assets and settings; normalized counts, edits, decisions, provenance; focused request tests | T-00 | Complete; 27 UI API tests and live HTTP phrase check passed |
| T-06 | Unified UI / Antigravity | Third mode in existing visual system; semantic controls, examples, annotated before/after, audit and estimate; responsive layout | T-05 response contract | Complete; semantic B0/B2 browser views, narrow layout, and JS syntax verified |
| T-07 | Retire duplicate dashboard / Orchestrator | Remove Streamlit-only entry point/dependencies/tests, update docs to single launch command, verify all modes | T-05, T-06 | Complete; tree and semantic browser runs passed |
| T-08 | Cross-vendor review and release check / Antigravity and Codex | Other vendor reviews changed code; relevant tests and live browser/API check pass | T-05, T-06, T-07 | Complete; review findings fixed, 567 tests passed, WordNet concurrency and desktop/narrow browser flows verified |

## Production-value evidence loop (2026-11-11)

Outcome: test real trained vectors against a frozen, representative workload and publish a measured go/no-go decision. Production acceptance requires measurable incremental savings over the simpler baseline, acceptable prompt-level meaning and task quality on held-out inputs, useful runtime/economics, and an explicit review/rollback workflow. Tooling completion alone does not satisfy this gate. No mentor or human judgment is inferred from software output.

| ID | Title / owner | Acceptance criterion | Depends on | Status |
| --- | --- | --- | --- | --- |
| T-09 | Real vectors and corpus feasibility / Antigravity | Identify licensed trained vector source, acquisition size/provenance, suitable frozen corpus and task labels; report verified feasibility | — | Complete; GloVe PDDL mirror and Dolly CC BY-SA prompt corpus identified; AG News license unknown and retained only for local task proxy |
| T-10 | Production evaluation gap analysis / Codex | Map current code and identify minimum paired, held-out, human-review, and economics gates with testable work items | — | Complete; read-only gap report in session f0e08b560e37416bad31c4d014cbd549 |
| T-11 | Freeze real-data pilot / Orchestrator | Obtain licensed local assets outside Git; create reproducible selected manifest; run U/B0/B1/B2 with real vectors and honest provenance | T-09 | Complete for two public proxies; 400 Dolly prompts and 400 AG News articles, local GloVe 50d, all four arms, hashes and split labels recorded; actual customer workload remains external |
| T-12 | Paired production decision report / Codex | Report by split and workload slice, incremental embedding value and uncertainty, latency/cost; separate observed quality from pending human review | T-10, T-11 | Complete; both frozen runs validated, held-out paired intervals reported, 8 focused tests passed |
| T-13 | Reviewed template workflow / Antigravity, finished by orchestrator | Permit explicit per-edit approval/rejection, exact-content cache invalidation and audit export in the unified UI/API; no automatic production rewrite | T-10 | Complete for local loopback use; browser review/save/export and stale-source invalidation verified |
| T-14 | Independent cross-vendor review and final gate / Orchestrator | Review each changed component with other vendor, verify tests and demo, document justified production recommendation and remaining external evidence | T-11, T-12, T-13, T-16 | Complete for local experimental release; 609 tests, browser/API approval demo and cross-vendor review pass; automatic rewriting remains a measured no-go |
| T-15 | AG News paired classification proxy / Codex | Offline deterministic model compares original and edited held-out news articles with known labels; mark as task proxy, not LLM quality | T-11 | Complete; 21 focused tests, actual 20,000-train/200-held-out run, 7 ambiguous hashes excluded; original and B0/B1/B2 all 91% topic accuracy |
| T-16 | Harden approval bypasses / Codex, finished by orchestrator after runner limit | Save only complete, authentic, reviewed server suggestions; validate saved templates against active config; block foreign Host; protect UI against stale review; adversarial tests | T-13, T-14 findings | Complete; 609 full-suite tests and live API/browser approval flow pass |
| T-17 | Final hardened approval review / Antigravity | Read-only cross-vendor review of T-16 security and stale-state fixes; reproduce any remaining high-severity issue | T-16 | Complete; session 35f1a3c1cbe541f289ba84df945e52a8 found no high-severity bug; two minor findings fixed and retested in root session |

## Full-vocabulary scaling (2026-11-25)

Outcome: make the existing locally provisioned 400,000-word GloVe asset usable in the shared workbench, retain exact deterministic cosine results, and compare vocabulary sizes on frozen prompts before making a quality claim. No large vector asset is committed.

| ID | Title / owner | Acceptance criterion | Depends on | Status |
| --- | --- | --- | --- | --- |
| T-18 | Exact top-k retrieval / Orchestrator | Replace full Python sort with bounded exact selection; preserve lexical ties, exclusions, and returned neighbors; focused equivalence tests | — | Complete; parity test covers ties, exclusions, and case variants |
| T-19 | Configurable workbench vocabulary / Orchestrator (reassigned after Antigravity CLI error) | Expose 10k/50k/400k local retention with honest loaded/source counts and warning for full-size cold load; preserve defaults and synthetic fixture labeling | — | Complete; API and focused UI tests pass |
| T-20 | Size-sweep reporter / Codex | Compare frozen independent 10k/50k/400k runs using existing manifests; report coverage, savings, time and memory honestly; no overwrite | — | Complete; report generated from three validated frozen runs, session 5f9a64425a4046b4a2d0ce0dd872669b |
| T-21 | Full GloVe pilot and review / Orchestrator | Run size sweep on the same frozen corpus, cross-vendor review code, verify tests and browser, document measured recommendation | T-18, T-19, T-20 | Complete; frozen size report, 628 tests, full-vocabulary browser demo, and independent Antigravity review (session a26546eca6ac480ba27923f1a3aa40fd) passed; automatic rewriting remains no-go |
