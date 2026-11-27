# Guarded LLMLingua-2

The protection rules can also guide a more aggressive compressor. Here is an
illustrative pass through the `SentinelHybridCompressor`:

```text
Original:       The team worked carefully, unless spending exceeds $5,000.
Protected:      unless spending exceeds $5,000
Sent to model:  The team worked carefully, KEEPCLAUSEA.
Model may trim: The team worked KEEPCLAUSEA.
Restored:       The team worked unless spending exceeds $5,000.
```

The conditional clause is hidden from LLMLingua-2, then restored exactly. The
hybrid also requires the main subject and verb to survive. If the model drops or
moves the marker, loses those anchors, or fails the protected-content check, the
sentence falls back to `promptcomp`'s conservative compression. A protected main
clause takes that conservative path directly; sentences without protected content
can go to LLMLingua-2. This is a guardrail for specific protected details, not a
guarantee that every nuance survives model compression.

Install the optional backend with:

```bash
python -m pip install -e ".[llmlingua]"
```

[`eval/hybrid.py`](../eval/hybrid.py) contains two research prototypes:

- `HybridCompressor` sends protected sentences through the conservative compressor
  and clean sentences through LLMLingua-2.
- `SentinelHybridCompressor` masks isolatable protected clauses, forces their
  sentinels and main-clause anchors through LLMLingua-2, restores the original clauses
  byte-for-byte, and falls back conservatively when validation fails.

```python
from eval.hybrid import SentinelHybridCompressor

compressor = SentinelHybridCompressor()
result = compressor.compress(text, 0.5)
print(result.text)
print(compressor.stats)
```

These hybrids compress further on some prose, but they are not part of the public API.
Protected-token validation is not the same as proof of semantic equivalence; use them
only for evaluation until downstream answer retention is measured.
