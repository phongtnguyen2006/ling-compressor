# Carryover for Plan B2 (M5 protect) and Plan B3 (M6 delete)

## MUST fix in B3 (before deletion is wired) — from B1 final review
- **Non-projective subtree guard.** `enumerate_candidates` sets char span = min/max token
  offset over `token.subtree` and `text = doc.text[start:end]`. For a NON-PROJECTIVE parse
  the subtree token set is not contiguous, so that slice includes intervening non-subtree
  tokens (and `n_tokens` < tokens actually in the slice). Deleting that slice could remove a
  core argument, violating "whole subtrees only / grammatical by construction" — the product's
  core guarantee. Before B3 deletes: detect discontinuity (e.g. compare `n_tokens` to the count
  of tokens whose idx falls in [start,end), or check projectivity) and skip/split such candidates.
  Add a regression test with a non-projective sentence.

## MUST honor in B2 (protect veto)
- The veto needs the SAME `Doc` the candidates came from: NER classes (ORG/MONEY/DATE/…) live on
  `doc.ents`; the negation/modal SCOPE veto needs the dependency tree (descendant-of-a-neg-child,
  or descendant of a modal aux). `Candidate` alone is insufficient. Signature should be
  `veto(doc, candidates, protect_list, ...)`. Recover a candidate's tokens via
  `doc.char_span(c.char_start, c.char_end)` — ranges are token-aligned so this returns a real Span.
- Share protected-token EXTRACTION logic between protect.py and the invariant test
  (`test_protect_invariants.py`) so "no protected token deleted" holds by construction (test and
  veto must agree on what counts as a number/negation/etc).

## Optional / nice
- Add a test asserting `original[start:start+len(c.text)] == c.text` for the inspect offset mapping.
- Wheel-build smoke check to prove packaging for the non-editable path (tests only cover editable).
- config_hash should include scorer identity once B3 makes deletion scorer-driven (from Plan A review).

## Confirmed sound after B2 (context for B3 deletion)
- protect veto now covers: numbers (regex+NER), NER entities, negation (dep+lemmas+n't),
  single-word terms, multi-word phrases (char-span OVERLAP), defined terms (yaml+auto, overlap),
  nested modal auxiliaries (span tokens), neg/modal scope. `veto()` also vetoes any candidate whose
  range doesn't token-align (`doc.char_span` None => "unaligned").
- B3 deletion must: only delete SURVIVORS from veto(); still add the non-projective guard (a survivor
  whose contiguous slice includes non-subtree tokens must be skipped/split before deletion).
- The invariant suite (B3) MUST reuse extract_numbers/extract_negations from protect.py (shared source of truth).

## M7 eval observations (from B3)
- Modal/neg scope veto propagates across conjunct clauses (conj attaches to first conjunct whose
  head bears the modal aux) -> a sentence with one "shall/must/may" can over-protect all its clauses.
  SAFE (over-protect) per spec 11.1, but expect LOW compression on obligation-heavy legal text.
  M7 should measure this; if compression is near-zero on the target domain, consider a tighter scope
  rule (clause-local rather than ancestor-chain) — but ONLY with invariant tests proving no under-veto.
- Re-add per-stage timings (segment/parse/veto/score/delete/substitute) to pipeline.timings_ms for
  the inspect "per-stage timing" requirement (spec 5); dropped during the B3 rewrite.

## M7 RESULT (2026-07-27): llmlingua2 baseline measured — spec §7.4 hypothesis CONFIRMED
memo1 (1282 tok): ours 0.872 ratio / 0 violations. llmlingua2 0.724@r0.7 / 65 viol,
0.513@r0.5 / 92 viol, 0.319@r0.3 / 106 viol. stopword 0.755 / 22 viol.
llmlingua2 destroys (r0.5): must x10, not x9, all x7, any x6, shall x2, may x2,
no x2, unless, if, in the event, without + money (`$25.00`, `$185.00` x2, `$120.00`,
`$65.00`, `$55.00`...). Also mangles `$5,000` -> `$ 5, 000`.
NOT a context-length artifact: short docs (72/79 tok, under BERT's 512) still show 11-13 viol.
=> The differentiator is real and strong. Compression gap (0.87 vs 0.32) is the honest cost;
   whether it is worth paying depends on accuracy retention, still unmeasured (needs API key).
=> Strengthens the case for the hybrid (route protected sentences to ours, rest to llmlingua2).
Fix required: LLMLingua defaults device_map=cuda -> raises on Apple Silicon, baseline was
silently skipped by available_baselines()'s except. Now auto-detects cuda->mps->cpu (9c25bb4).
