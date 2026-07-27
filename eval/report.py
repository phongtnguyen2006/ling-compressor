"""Render eval results to CSV and markdown."""
from __future__ import annotations

import csv
import io
from dataclasses import asdict, fields

from .harness import RunResult

_HEADER = [f.name for f in fields(RunResult)]


def to_csv(results: list[RunResult]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_HEADER)
    writer.writeheader()
    for r in results:
        writer.writerow(asdict(r))
    return buf.getvalue()


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def to_markdown(results: list[RunResult]) -> str:
    cols = ["compressor", "ratio", "tokens_before", "tokens_after", "compression_ratio",
            "protected_violation_count", "parse_failure_rate", "accuracy_retention", "latency_ms"]
    # accuracy_retention vs the 'none' ceiling per (doc, ratio)
    ceiling = {
        (r.doc_id, r.ratio): r.accuracy_f1
        for r in results if r.compressor == "none" and r.accuracy_f1 is not None
    }
    lines = ["# Eval report", "",
             "`protected_violation_count` must be **0** for `ours` (nonzero = build failure).",
             "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in sorted(results, key=lambda x: (x.doc_id, x.compressor, x.ratio)):
        base = ceiling.get((r.doc_id, r.ratio))
        retention = (r.accuracy_f1 / base) if (base and r.accuracy_f1 is not None) else None
        row = {
            "compressor": r.compressor, "ratio": r.ratio,
            "tokens_before": r.tokens_before, "tokens_after": r.tokens_after,
            "compression_ratio": r.compression_ratio,
            "protected_violation_count": r.protected_violation_count,
            "parse_failure_rate": r.parse_failure_rate,
            "accuracy_retention": retention, "latency_ms": r.latency_ms,
        }
        lines.append("| " + " | ".join(_fmt(row[c]) for c in cols) + " |")
    return "\n".join(lines)
