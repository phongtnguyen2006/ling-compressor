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
