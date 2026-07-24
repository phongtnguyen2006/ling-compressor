"""Inspection harness: paste/feed text, see the compression result.

Usage:
    python scripts/inspect.py --file data/samples/memo1.txt --ratio 0.6
    python scripts/inspect.py --stdin --out result.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

# Make the src-layout package importable when run as a loose script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from promptcomp import compress  # noqa: E402


def _result_to_dict(result) -> dict:
    d = dataclasses.asdict(result)
    return d


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect compression output.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", type=str, help="Path to a text file to compress.")
    src.add_argument("--stdin", action="store_true", help="Read text from stdin.")
    parser.add_argument("--ratio", type=float, default=0.6, help="Target ratio.")
    parser.add_argument("--min-tokens", type=int, default=500, help="Size gate.")
    parser.add_argument("--no-substitute", action="store_true", help="Disable Stage 4.")
    parser.add_argument("--out", type=str, default=None, help="Write JSON result here.")
    parser.add_argument("--cache-path", type=str, default=None, help="Token cache db.")
    args = parser.parse_args(argv)

    if args.stdin:
        text = sys.stdin.read()
    else:
        text = Path(args.file).read_text()

    result = compress(
        text,
        target_ratio=args.ratio,
        substitute=not args.no_substitute,
        min_tokens=args.min_tokens,
        cache_path=args.cache_path,
    )

    print(f"token counter : {result.token_counter}")
    print(f"tokens before : {result.tokens_before}")
    print(f"tokens after  : {result.tokens_after}")
    print(f"ratio         : {result.ratio:.3f}")
    print(f"timings (ms)  : {result.timings_ms}")
    if result.substitutions:
        print("substitutions :")
        for k, v in result.substitutions.items():
            print(f"    {k!r} -> {v!r}")
    if result.warnings:
        print("warnings      :")
        for w in result.warnings:
            print(f"    - {w}")

    if args.out:
        Path(args.out).write_text(json.dumps(_result_to_dict(result), indent=2))
        print(f"wrote         : {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
