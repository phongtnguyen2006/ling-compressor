# Plan B carryover (from Plan A final review)

Deferred findings to address during Plan B (M4-M6):

## Important
- **Package the data files.** `src/promptcomp/substitute.py` loads `data/substitutions.yaml`
  via `Path(__file__).resolve().parents[2] / "data"`, which only exists in an editable
  checkout. A built wheel omits `data/`, so `compress()` raises FileNotFoundError above the
  size gate. Fix: move YAML (and later protect_terms.yaml, defined_terms.yaml) under
  `src/promptcomp/data/` and load via `importlib.resources`, or declare setuptools package-data.

## Minor / design notes
- **config_hash must include scorer identity** once Plan B makes deletion scorer-driven —
  otherwise two runs with different scorers collide on the same hash despite different output.
- **Substitution dictionary validation**: `\b` boundaries are unsound for phrase keys that
  start/end in non-word chars, and sequential in-place replacement can re-substitute text a
  longer key inserted if a replacement value collides with another key. None of the shipped
  16 phrases trigger either. Add a load-time validation check before growing the dictionary
  (esp. when entity-abbreviation lands).
- **Sentence-initial casing**: phrase shortening lowercases sentence-initial phrases
  ("In order to" -> "to"). Baked into a Plan A test; revisit for grammatical output.
- **Hermetic anthropic test**: `test_anthropic_counter_missing_dep_is_actionable` relies on
  `anthropic` being absent; monkeypatch the import so CI with the extra installed stays hermetic.
- **inspect.py I/O guarding**: missing --file / unwritable --out raise tracebacks rather than
  a clean nonzero exit.
