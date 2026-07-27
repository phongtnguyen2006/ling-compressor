from eval.report import to_csv, to_markdown
from eval.harness import RunResult


def _row(name, viol):
    return RunResult(
        doc_id="d1", compressor=name, ratio=0.5, tokens_before=100, tokens_after=70,
        compression_ratio=0.7, protected_violation_count=viol, latency_ms=1.2,
        parse_failure_rate=0.0,
        accuracy_exact=None, accuracy_f1=None, accuracy_judge=None,
    )


def test_csv_has_header_and_rows():
    csv = to_csv([_row("ours", 0), _row("stopword", 3)])
    lines = csv.strip().splitlines()
    assert lines[0].startswith("doc_id,compressor,ratio")
    assert len(lines) == 3
    assert "ours" in csv and "stopword" in csv


def test_markdown_flags_violations():
    md = to_markdown([_row("ours", 0), _row("stopword", 3)])
    assert "protected_violation_count" in md
    assert "| ours |" in md or "ours" in md
