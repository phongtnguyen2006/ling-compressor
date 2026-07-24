"""CLI: run the offline eval grid over a corpus and write reports.

    python eval/run.py --corpus data/samples --out eval/results --ratios 0.3,0.5,0.7
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.baselines import available_baselines  # noqa: E402
from eval.harness import run_grid  # noqa: E402
from eval.report import to_csv, to_markdown  # noqa: E402
from promptcomp.tokens import CachedCounter, TiktokenCounter  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the offline compression eval grid.")
    parser.add_argument("--corpus", default="data/samples", help="Dir of *.txt documents.")
    parser.add_argument("--out", default="eval/results", help="Output dir for reports.")
    parser.add_argument("--ratios", default="0.3,0.5,0.7", help="Comma-separated target ratios.")
    parser.add_argument("--cache-path", default=None, help="Token cache db.")
    args = parser.parse_args(argv)

    corpus_dir = Path(args.corpus)
    docs = {p.stem: p.read_text() for p in sorted(corpus_dir.glob("*.txt"))}
    if not docs:
        print(f"no *.txt documents found in {corpus_dir}")
        return 1

    ratios = [float(r) for r in args.ratios.split(",")]
    counter = CachedCounter(TiktokenCounter(), args.cache_path or ".token_cache.sqlite")

    results = run_grid(docs, available_baselines(), ratios, counter)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.md").write_text(to_markdown(results))
    (out_dir / "report.csv").write_text(to_csv(results))

    ours_violations = sum(
        r.protected_violation_count for r in results if r.compressor == "ours"
    )
    print(f"docs={len(docs)} compressors={len({r.compressor for r in results})} "
          f"ratios={len(ratios)} rows={len(results)}")
    print(f"ours protected_violation_count total = {ours_violations} (must be 0)")
    print(f"wrote {out_dir/'report.md'} and {out_dir/'report.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
