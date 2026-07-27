"""Orchestrate documents × compressors × ratios → RunResult rows. Offline by default."""
from __future__ import annotations

from dataclasses import dataclass

from .metrics import compression_metrics, parse_failure_rate, protected_violation_count


@dataclass(frozen=True)
class RunResult:
    doc_id: str
    compressor: str
    ratio: float
    tokens_before: int
    tokens_after: int
    compression_ratio: float
    protected_violation_count: int
    latency_ms: float
    parse_failure_rate: float
    accuracy_exact: float | None
    accuracy_f1: float | None
    accuracy_judge: float | None


def run_grid(docs, compressors, ratios, counter, grader=None) -> list[RunResult]:
    results: list[RunResult] = []
    for doc_id, text in docs.items():
        for compressor in compressors:
            for ratio in ratios:
                out = compressor.compress(text, ratio)
                m = compression_metrics(text, out.text, counter)
                violations = protected_violation_count(text, out.text)
                pfr = parse_failure_rate(out.text)
                acc = grader(doc_id, out.text) if grader is not None else None
                results.append(
                    RunResult(
                        doc_id=doc_id,
                        compressor=compressor.name,
                        ratio=ratio,
                        tokens_before=m["tokens_before"],
                        tokens_after=m["tokens_after"],
                        compression_ratio=m["compression_ratio"],
                        protected_violation_count=violations,
                        latency_ms=round(out.latency_ms, 3),
                        parse_failure_rate=round(pfr, 4),
                        accuracy_exact=(acc or {}).get("exact") if acc else None,
                        accuracy_f1=(acc or {}).get("f1") if acc else None,
                        accuracy_judge=(acc or {}).get("judge") if acc else None,
                    )
                )
    return results
