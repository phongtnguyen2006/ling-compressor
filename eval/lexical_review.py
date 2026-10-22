"""Validate human annotations against a frozen lexical pilot and report coverage.

Run ``python -m eval.lexical_review --help``. No model fills judgment fields.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

from .lexical_pilot import _digest, _review_exports, _write_json, _write_jsonl

JUDGMENTS = ("meaning_preserved", "grammatical", "harmful_change")
ANNOTATIONS = frozenset((*JUDGMENTS, "reviewer", "notes"))
RUN_FILES = frozenset({
    "manifest.json", "inputs.jsonl", "results.jsonl", "candidate-pools.jsonl",
    "summary.json", "summary.md", "edit-review.jsonl", "prompt-review.jsonl",
    "review-key.jsonl",
})


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"invalid JSON constant: {value}")


def _parse(text):
    return json.loads(text, object_pairs_hook=_object, parse_constant=_reject_constant)


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = _parse(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path.name}:{number}: expected a JSON object")
        rows.append(row)
    return rows


def _canonical(value) -> str:
    """Type-sensitive comparison, including booleans versus integers."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def _by_id(rows: list[dict]) -> dict[str, dict]:
    result = {}
    for row in rows:
        review_id = row.get("review_id")
        if not isinstance(review_id, str) or not review_id:
            raise ValueError("review_id must be a nonempty string")
        if review_id in result:
            raise ValueError(f"duplicate review_id: {review_id}")
        result[review_id] = row
    return result


def _merge_annotations(baseline: list[dict], submitted: Path | None) -> dict[str, dict]:
    originals = _by_id(baseline)
    merged = {review_id: dict(row) for review_id, row in originals.items()}
    if submitted is None:
        return merged
    for review_id, row in _by_id(_read_jsonl(submitted)).items():
        if review_id not in originals:
            raise ValueError(f"unknown review_id: {review_id}")
        original = originals[review_id]
        if row.keys() != original.keys():
            raise ValueError(f"review fields differ from frozen source: {review_id}")
        immutable = {key: value for key, value in row.items() if key not in ANNOTATIONS}
        expected = {key: value for key, value in original.items() if key not in ANNOTATIONS}
        if _canonical(immutable) != _canonical(expected):
            raise ValueError(f"immutable review content changed: {review_id}")
        for key in JUDGMENTS:
            if row[key] is not None and type(row[key]) is not bool:
                raise ValueError(f"{key} must be true, false, or null: {review_id}")
        reviewer = row["reviewer"]
        if reviewer is not None and (not isinstance(reviewer, str) or not reviewer.strip()):
            raise ValueError(f"reviewer must be null or a nonblank string: {review_id}")
        if any(row[key] is not None for key in JUDGMENTS) and reviewer is None:
            raise ValueError(f"judgments require a reviewer identity: {review_id}")
        if not isinstance(row["notes"], str):
            raise ValueError(f"notes must be a string: {review_id}")
        merged[review_id] = row
    return merged


def _state(row: dict) -> str:
    count = sum(row[key] is not None for key in JUDGMENTS)
    return "completed" if count == len(JUDGMENTS) else "partial" if count else "pending"


def _accepted(row: dict) -> bool:
    return (_state(row) == "completed" and row["meaning_preserved"] is True
            and row["grammatical"] is True and row["harmful_change"] is False)


def _coverage(rows: list[dict]) -> dict:
    completed = [row for row in rows if _state(row) == "completed"]
    return {
        "total": len(rows), "completed": len(completed),
        "partial": sum(_state(row) == "partial" for row in rows),
        "pending": sum(_state(row) == "pending" for row in rows),
        "accepted": sum(_accepted(row) for row in completed),
        "rejected": sum(not _accepted(row) for row in completed),
        "harmful_completed": sum(row["harmful_change"] is True for row in completed),
    }


def summarize_reviews(run_dir: Path, *, edit_reviews: Path | None = None,
                      prompt_reviews: Path | None = None) -> dict:
    """Validate a single annotation set. Omitted records remain unanswered.

    Edit and prompt files are submitted copies; the source run is never modified.
    Human identity is recorded, not authenticated. This is not multi-rater analysis.
    """
    run_dir = Path(run_dir)
    checksums = _parse((run_dir / "checksums.json").read_text(encoding="utf-8"))
    if not isinstance(checksums, dict) or set(checksums) != RUN_FILES:
        raise ValueError("frozen run must contain the complete lexical-pilot-v1 checksum set")
    for name, expected in checksums.items():
        actual = hashlib.sha256((run_dir / name).read_bytes()).hexdigest()
        if not isinstance(expected, str) or actual != expected:
            raise ValueError(f"frozen run checksum mismatch: {name}")
    manifest = _parse((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != "lexical-pilot-v1":
        raise ValueError("unsupported pilot schema")
    identity_keys = ("schema_version", "inputs", "counter", "providers", "configuration",
                     "versions", "software_sha256")
    if manifest.get("run_fingerprint") != _digest({key: manifest[key] for key in identity_keys}):
        raise ValueError("frozen manifest fingerprint is inconsistent")
    results = _read_jsonl(run_dir / "results.jsonl")
    seen = set()
    for result in results:
        identity = (result["doc_id"], result["arm"])
        if identity in seen:
            raise ValueError("duplicate document/arm in frozen results")
        seen.add(identity)
        counts = (result["tokens_before"], result["tokens_after"])
        if any(type(count) is not int or count < 0 for count in counts) or counts[1] > counts[0]:
            raise ValueError("invalid complete-output token counts in frozen results")
    if not results:
        raise ValueError("frozen run has no results")
    baseline_edits = _read_jsonl(run_dir / "edit-review.jsonl")
    baseline_prompts = _read_jsonl(run_dir / "prompt-review.jsonl")
    keys = _read_jsonl(run_dir / "review-key.jsonl")
    # Detect an inconsistent review key even if somebody recalculated checksums.
    expected_exports = _review_exports(results)
    for actual, expected in zip((baseline_edits, baseline_prompts, keys), expected_exports):
        if _canonical(actual) != _canonical(expected):
            raise ValueError("frozen review exports do not match occurrence results")
    edits = _merge_annotations(baseline_edits, edit_reviews)
    prompts = _merge_annotations(baseline_prompts, prompt_reviews)
    summaries = []
    for arm in dict.fromkeys(row["arm"] for row in results):
        arm_results = [row for row in results if row["arm"] == arm]
        changed, unchanged, reviewed_savings = [], [], 0
        for result in arm_results:
            review_id = _digest({"doc_id": result["doc_id"], "original": result["original"],
                                 "suggested": result["suggested"]})
            annotation = prompts[review_id]
            if result["original"] == result["suggested"]:
                unchanged.append(annotation)
            else:
                changed.append(annotation)
                if _accepted(annotation):
                    reviewed_savings += result["tokens_before"] - result["tokens_after"]
        arm_edit_keys = [key for key in keys if key["arm"] == arm and key["kind"] == "edit"]
        candidate_ids = {key["review_id"] for key in arm_edit_keys}
        selected_ids = {key["review_id"] for key in arm_edit_keys if key["decision"] == "selected"}
        before = sum(row["tokens_before"] for row in arm_results)
        summaries.append({
            "arm": arm, "documents": len(arm_results), "changed_documents": len(changed),
            "tokens_before": before,
            "proposed_tokens_saved": sum(row["tokens_before"] - row["tokens_after"] for row in arm_results),
            "changed_prompt_reviews": _coverage(changed),
            "unchanged_prompt_reviews": _coverage(unchanged),
            "selected_edit_reviews": _coverage([edits[key] for key in sorted(selected_ids)]),
            "candidate_edit_reviews": _coverage([edits[key] for key in sorted(candidate_ids)]),
            "tokens_saved_in_review_accepted_outputs": reviewed_savings,
            "review_accepted_savings_over_all_input_tokens": reviewed_savings / before if before else 0.0,
        })
    submitted = {kind: {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                 if path is not None else None
                 for kind, path in (("edit", edit_reviews), ("prompt", prompt_reviews))}
    identity = {"schema_version": "lexical-review-v1", "run_fingerprint": manifest["run_fingerprint"],
                "source_checksums": checksums, "submitted": submitted,
                "reporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    return {
        **identity, "report_fingerprint": _digest(identity),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "annotation_summary_not_deployment_approval",
        "unique_edit_reviews": _coverage(list(edits.values())),
        "unique_prompt_reviews": _coverage(list(prompts.values())),
        "arms": summaries,
        "limitations": [
            "Judgments are supplied annotations; reviewer identity is not authenticated.",
            "Incomplete and omitted reviews are not passes. Candidate ratings cannot approve composed outputs.",
            "Shared review IDs are deduplicated within each arm; arms and repeated templates are not independent samples.",
            "Changed outputs and unchanged controls have separate denominators; no population error rate is estimated.",
            "Review-accepted savings count only fully accepted changed outputs against all input tokens; this is not net API savings.",
            "One annotation set is supported; no rater agreement, sampling weighting, confidence interval, or downstream evaluation is inferred.",
            "Checksums bind artifacts for reproducibility and accidental-edit detection; they are not signed proof of authenticity.",
        ],
        "annotations": {"edits": list(edits.values()), "prompts": list(prompts.values())},
    }


def write_report(report: dict, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError("review output must be a new or empty directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "report.json", {key: value for key, value in report.items() if key != "annotations"})
    _write_jsonl(output_dir / "edit-annotations.jsonl", report["annotations"]["edits"])
    _write_jsonl(output_dir / "prompt-annotations.jsonl", report["annotations"]["prompts"])
    lines = ["---", f"id: SEM-REVIEW-{report['report_fingerprint'][:12]}", "type: review-evidence",
             "status: annotation-summary", "tags: [semantic-compression, review]", "---", "",
             "# Lexical review coverage", "", f"Source run: `{report['run_fingerprint']}`", "",
             "Annotation status only; this report does not approve deployment.", "",
             "| Arm | Changed outputs reviewed | Accepted / rejected | Unchanged controls reviewed | Tokens saved in accepted outputs |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for row in report["arms"]:
        changed, unchanged = row["changed_prompt_reviews"], row["unchanged_prompt_reviews"]
        lines.append(f"| {row['arm']} | {changed['completed']}/{changed['total']} | "
                     f"{changed['accepted']} / {changed['rejected']} | {unchanged['completed']}/{unchanged['total']} | "
                     f"{row['tokens_saved_in_review_accepted_outputs']} |")
    lines.extend(["", "Pending and partial judgments remain in the [report data](report.json).", "",
                  "- [Edit annotations](edit-annotations.jsonl)",
                  "- [Complete-prompt annotations](prompt-annotations.jsonl)", "", "## Limitations", ""])
    lines.extend(f"- {item}" for item in report["limitations"])
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_json(output_dir / "checksums.json", {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output_dir.iterdir()) if path.is_file()
    })


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Original frozen pilot directory")
    parser.add_argument("--edit-reviews", type=Path, help="Annotated copy; omitted rows stay pending")
    parser.add_argument("--prompt-reviews", type=Path, help="Annotated copy; omitted rows stay pending")
    parser.add_argument("--out", type=Path, required=True, help="New report directory")
    args = parser.parse_args(argv)
    try:
        report = summarize_reviews(args.run, edit_reviews=args.edit_reviews, prompt_reviews=args.prompt_reviews)
        write_report(report, args.out)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"lexical review: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {args.out / 'summary.md'}; "
          f"{report['unique_prompt_reviews']['completed']}/{report['unique_prompt_reviews']['total']} "
          "unique complete-prompt reviews completed. No deployment approval inferred.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
