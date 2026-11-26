"""Create two independent blank reviewer packets from a frozen lexical pilot.

Run ``python -m eval.lexical_review_packet --help``. No judgments are created, and
reviewer files carry no arm, provider, selection, or token-savings information.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

from .lexical_pilot import _write_json, _write_jsonl
from .lexical_review import _read_jsonl, summarize_reviews

SCHEMA = "lexical-review-packet-v1"
REVIEWERS = ("reviewer-a", "reviewer-b")
DEFAULT_CONTROLS = 20


def _rank(label: str, review_id: str) -> str:
    return hashlib.sha256(f"{SCHEMA}\0{label}\0{review_id}".encode("utf-8")).hexdigest()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_packet(run_dir: Path, output_dir: Path, *, control_count: int | None = None) -> dict:
    """Validate the frozen run, then write blank reviewer-a/b JSONL files to a new directory."""
    run_dir, output_dir = Path(run_dir), Path(output_dir)
    if control_count is not None and (type(control_count) is not int or control_count < 0):
        raise ValueError("control count must be nonnegative")
    if output_dir.resolve().is_relative_to(run_dir.resolve()):
        raise ValueError("packet output cannot be placed inside the frozen run")
    if output_dir.exists():
        raise ValueError("packet output must be a new path; existing files are not overwritten")
    report = summarize_reviews(run_dir)  # checksums, schema, and export consistency
    rows = _read_jsonl(run_dir / "prompt-review.jsonl")
    changed = [row for row in rows if row["original"] != row["suggested"]]
    changed_docs = {row["doc_id"] for row in changed}
    # Unchanged arm outputs of a changed document would show reviewers the same prompt twice.
    unchanged = [row for row in rows if row["original"] == row["suggested"]
                 and row["doc_id"] not in changed_docs]
    if not changed:
        raise ValueError("frozen run has no changed complete-prompt records to review")
    wanted = min(len(changed), DEFAULT_CONTROLS) if control_count is None else control_count
    if wanted > len(unchanged):
        if control_count is not None:
            raise ValueError(f"requested {control_count} controls but only {len(unchanged)} unchanged records exist")
        wanted = len(unchanged)
    controls = sorted(unchanged, key=lambda row: _rank("control", row["review_id"]))[:wanted]
    selected = changed + controls
    output_dir.mkdir(parents=True)
    files = {}
    for label in REVIEWERS:
        ordered = sorted(selected, key=lambda row: _rank(label, row["review_id"]))
        _write_jsonl(output_dir / f"{label}.jsonl", [dict(row) for row in ordered])
        files[f"{label}.jsonl"] = {"records": len(ordered), "sha256": _sha(output_dir / f"{label}.jsonl")}
    metadata = {
        "schema_version": SCHEMA, "run_fingerprint": report["run_fingerprint"],
        "source_checksums": report["source_checksums"],
        "source_prompt_review_sha256": report["source_checksums"]["prompt-review.jsonl"],
        "counts": {"source_unique_prompts": len(rows), "changed": len(changed),
                   "unchanged_available": len(unchanged), "controls": len(controls),
                   "per_reviewer": len(selected)},
        "control_count_requested": control_count,
        "control_selection": "lowest sha256(schema, 'control', review_id) among unchanged records",
        "files": files,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "note": "Blank independent copies; judgments must be supplied by humans. "
                "Keep this metadata separate from reviewers; it reveals the changed/control split.",
    }
    _write_json(output_dir / "packet.json", metadata)
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Original frozen pilot directory")
    parser.add_argument("--out", type=Path, required=True, help="New packet directory")
    parser.add_argument("--control-count", type=int, help="Unchanged controls (default min(changed, 20))")
    args = parser.parse_args(argv)
    try:
        metadata = build_packet(args.run, args.out, control_count=args.control_count)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"lexical review packet: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({key: metadata[key] for key in ("run_fingerprint", "counts", "files")},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
